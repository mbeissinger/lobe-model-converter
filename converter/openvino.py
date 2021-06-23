"""
Converts a Lobe SavedModel into a frozen graph that the Intel Openvino converter can accept :)
"""
import argparse
from converter.compatibility import strip_incompatible_ops_dtypes


def convert_openvino(savedmodel_dir: str, export_path: str, reshape_for_percept=False):
	# just strip any string tensors for now, revisit the fully supported op list later
	strip_incompatible_ops_dtypes(savedmodel_dir, export_path, dtypes=["string"], reshape_for_percept=reshape_for_percept)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a frozen graph from a Lobe Tensorflow SavedModel export that is compatible with Intel Openvino.')
	parser.add_argument('savedmodel_dir', help='Path to the SavedModel directory.')
	parser.add_argument('export_path', help='Filepath for the exported frozen graph.')
	# add command line flag argument for reshaping the output for Azure Percept
	parser.add_argument('--azure-percept', dest='reshape_for_percept', action='store_true', help='Reshape the output to be compatible for Azure Percept.')
	parser.set_defaults(reshape_for_percept=False)
	args = parser.parse_args()
	convert_openvino(savedmodel_dir=args.savedmodel_dir, export_path=args.export_path, reshape_for_percept=args.reshape_for_percept)
