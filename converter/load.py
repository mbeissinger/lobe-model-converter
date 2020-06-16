"""
Loads models
"""
import os
import json
from typing import Tuple, Dict
import tensorflow as tf


def load_savedmodel(savedmodel_dir: str) -> Tuple[tf.compat.v1.Session, Dict[str, any]]:
	"""
	Loads a Lobe exported Tensorflow SavedModel and returns the session with the model loaded and our
	signature file.
	"""
	# make sure our exported SavedModel folder exists
	model_path = os.path.realpath(savedmodel_dir)
	if not os.path.exists(model_path):
		raise ValueError(f"Exported model folder doesn't exist {savedmodel_dir}")

	# load our signature json file, this shows us the model inputs and outputs
	# you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
	with open(os.path.join(model_path, "signature.json"), "r") as f:
		signature = json.load(f)

	# make the tensorflow session and load the model
	session = tf.compat.v1.Session(graph=tf.Graph())
	# load our model into the session
	tf.compat.v1.saved_model.load(sess=session, tags=signature.get("tags"), export_dir=model_path)

	return session, signature
