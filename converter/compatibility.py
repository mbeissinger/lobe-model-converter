"""
Load a Tensorflow SavedModel file from Lobe's export, and strip away commonly unsupported operations from other
frameworks.
"""
import os
import json
from typing import List
from converter.load import load_savedmodel
from tensorflow.python.tools.freeze_graph import freeze_graph
import tensorflow as tf


def strip_incompatible_ops_dtypes(savedmodel_dir: str, export_path: str, ops: List[str] = None, dtypes: List[str] = None):
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
	for key, tensor_name in out_tensor_names.items():
		# if this tensor doesn't depend on any of the listed ops or dtypes, add it to our outputs for freeze_graph
		if not tensor_dependency(graph=graph, name=tensor_name, ops=ops, dtypes=dtypes):
			pruned_out_tensor_names[key] = tensor_name

	if len(pruned_out_tensor_names) == 0:
		print(f"No compatible outputs found for the model. Pruned dtypes {dtypes}, pruned ops {ops}")
		return

	# freeze_graph expects a comma separated list of tensor names without the :0 selectors
	output_node_names = ','.join([name.split(':')[0] for name in pruned_out_tensor_names.values()])

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

	out_signature_filename = os.path.join(os.path.dirname(os.path.abspath(export_path)), "signature_frozen_graph.json")
	with open(out_signature_filename, 'w') as f:
		json.dump(signature, f)


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
