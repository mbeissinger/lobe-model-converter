"""
Load a Tensorflow SavedModel file from Lobe's export, and strip away commonly unsupported operations from other
frameworks.
"""
import os
import json
import shutil
import uuid
from typing import List
from converter.load import load_savedmodel
from tensorflow.python.tools.freeze_graph import freeze_graph
import tensorflow as tf


def strip_incompatible_ops_dtypes(savedmodel_dir: str, export_path: str, ops: List[str] = None, dtypes: List[str] = None, reshape_for_percept=False):
	"""
	Load a SavedModel, strip away any ops and dtypes, and save the resulting pruned model to the export_path as a
	tensorflow frozen graph.
	"""
	if ops is None:
		ops = list()
	if dtypes is None:
		dtypes = list()
	ops = [op.lower() for op in ops]
	dtypes = [dtype.lower() for dtype in dtypes]

	# load our savedmodel and lobe signature (gives us inputs, outputs, classes, etc. -- meta properties about the model)
	session, signature = load_savedmodel(savedmodel_dir=savedmodel_dir)
	# get our tf graph from the session and the list of output tensor names from the signature
	graph = session.graph
	# map the output tensors we want to consider for this model -- prune any that are already in the dtype prune list
	out_tensor_names = {
		key: val.get("name") for key, val in signature.get("outputs", {}).items()
		if val.get("dtype", "").lower() not in dtypes
	}

	# if we already pruned all the model outputs, we are out of luck :(
	if len(out_tensor_names) == 0:
		print(f"No compatible outputs found for the model. Pruned dtypes {dtypes}")
		return

	# now traverse the tensorflow graph starting at the outputs and prune the output if it depends on any of the
	# listed dtypes or ops
	pruned_out_tensor_names = dict()
	new_outs = False
	pruned_out_shapes = dict()
	for key, tensor_name in out_tensor_names.items():
		# if this tensor doesn't depend on any of the listed ops or dtypes, add it to our outputs for freeze_graph
		if not tensor_dependency(graph=graph, name=tensor_name, ops=ops, dtypes=dtypes):
			pruned_out_tensor_names[key] = tensor_name

		# if this tensor has shape [None, classes], reshape it to [None, 1, 1, classes] (if we want to reshape it for Azure Percept)
		tensor = graph.get_tensor_by_name(tensor_name)
		if tensor.shape.as_list() == [None, len(signature.get("classes", {}).get("Label", []))] and reshape_for_percept:
			with graph.as_default():
				reshaped_out = tf.reshape(tensor, [-1, 1, 1, tensor.shape.as_list()[-1]])
			pruned_out_tensor_names[key] = reshaped_out.name
			pruned_out_shapes[key] = reshaped_out.shape.as_list()
			new_outs = True

	if len(pruned_out_tensor_names) == 0:
		print(f"No compatible outputs found for the model. Pruned dtypes {dtypes}, pruned ops {ops}")
		return

	if new_outs:
		with graph.as_default():
			input_sigs = {tensor_name.split(':')[0]:
				tf.compat.v1.saved_model.utils.build_tensor_info(
					graph.get_tensor_by_name(tensor_name)
				) for tensor_name in [val.get('name') for val in signature.get('inputs', {}).values()]
			}
			output_sigs = {tensor_name.split(':')[0]:
				tf.compat.v1.saved_model.utils.build_tensor_info(
					graph.get_tensor_by_name(tensor_name)
				) for tensor_name in list(pruned_out_tensor_names.values())
			}
			prediction_signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
				inputs=input_sigs, outputs=output_sigs, method_name=tf.saved_model.PREDICT_METHOD_NAME
			)
			meta_kwargs = {
				"sess": session,
				"tags": signature.get('tags'),
				"signature_def_map": {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
			}
			savedmodel_dir = os.path.join(savedmodel_dir, f"reshaped_savedmodel_{uuid.uuid4()}")
			print(f"Saving new model to {savedmodel_dir}")
			builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(savedmodel_dir)
			builder.add_meta_graph_and_variables(**meta_kwargs)
			builder.save()

	# freeze_graph expects a comma separated list of tensor names without the :0 selectors
	output_node_names = ','.join([name.split(':')[0] for name in pruned_out_tensor_names.values()])
	print(f"Using outs: {output_node_names}")

	# freeze the graph! this prunes anything not used to create the output node names
	freeze_graph(
		input_graph=None,
		input_saver=False,
		input_binary=False,
		input_checkpoint=None,
		output_node_names=output_node_names,
		restore_op_name=None,
		filename_tensor_name=None,
		output_graph=export_path,
		clear_devices=True,
		initializer_nodes="",
		input_saved_model_dir=savedmodel_dir,
		saved_model_tags=','.join(signature.get('tags'))
	)

	# make the signature json reflect the pruned outputs
	for out_key in list(signature.get("outputs", {}).keys()):
		if out_key not in pruned_out_tensor_names:
			del signature.get("outputs", {})[out_key]
		# otherwise if we reshaped the output, update the tensor name and shape
		elif new_outs and out_key in pruned_out_shapes:
			signature.get("outputs", {})[out_key]["shape"] = pruned_out_shapes[out_key]
			signature.get("outputs", {})[out_key]["name"] = pruned_out_tensor_names[out_key]
	out_signature_filename = os.path.join(os.path.dirname(os.path.abspath(export_path)), "signature_frozen_graph.json")
	with open(out_signature_filename, 'w') as f:
		json.dump(signature, f)

	# cleanup -- if we created a new saved model for the reshaped outputs, delete the saved model directory
	if new_outs:
		print(f"Removing temp saved model {savedmodel_dir}")
		shutil.rmtree(savedmodel_dir)


def tensor_dependency(graph: tf.Graph, name: str, ops: List[str], dtypes: List[str]):
	"""
	Given a Tensorflow graph, a tensor name in the graph, and list of ops and dtypes to prune, return if this
	tensor depends on any of the given ops and dtypes.

	Recursive search over the graph starting from this tensor to determine dependency on any of the ops or dtypes.
	"""
	tensor = graph.get_tensor_by_name(name)
	# check if this tensor depends on any of the listed dtypes, or if the op that created it is in the list of ops
	if tensor.dtype.name.lower() in dtypes or tensor.op.type.lower() in ops:
		return True

	# if this tensor's op has inputs, traverse the graph to see if it depends on any of the dtypes or ops
	for op_input in tensor.op.inputs:
		if tensor_dependency(graph, op_input.name, ops, dtypes):
			return True

	# otherwise return false, it doesn't depend on any of the listed ops or dtypes :)
	return False
