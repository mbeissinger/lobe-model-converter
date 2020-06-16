"""
Converts a Lobe SavedModel into a frozen graph that the Intel Openvino converter can accept :)
"""
import argparse
from converter.compatibility import strip_incompatible_ops_dtypes


def convert_openvino(savedmodel_dir: str, export_path: str):
	# just strip any string tensors for now, revisit the fully supported op list later
	strip_incompatible_ops_dtypes(savedmodel_dir, export_path, dtypes=["string"])


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create a frozen graph from a Lobe Tensorflow SavedModel export that is compatible with Intel Openvino.')
	parser.add_argument('savedmodel_dir', help='Path to the SavedModel directory.')
	parser.add_argument('export_path', help='Filepath for the exported frozen graph.')
	args = parser.parse_args()
	convert_openvino(savedmodel_dir=args.savedmodel_dir, export_path=args.export_path)
