# 05-upload-data.py
from azureml.core import Workspace
import argparse

ws = Workspace.from_config()
datastore = ws.get_default_datastore()

datastore.upload(src_dir='./data',
                    target_path='datasets/cifar10',
                    overwrite=True)

"""if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        default=None,
        help='Path to the local data directory'
    )

    parser.add_argument(
        '--target_path',
        type=str,
        default=None,
        help='Target path name'
    )

    args = parser.parse_args()
    datastore.upload(src_dir=args.src_dir,
                    target_path=parser.target_path,
                    overwrite=True)"""