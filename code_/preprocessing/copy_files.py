import argparse
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description="Copy edf files to one folder")
parser.add_argument("input_path", help="Path to the root folder of the EEG dataset")
parser.add_argument("export_path", help="Path where the output should be stored")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-ar", action="store_true", help="Take the tcp_ar dataset")
group.add_argument("-le", action="store_true", help="Take the tcp_le dataset")
args = parser.parse_args()


if args.ar:
    list_path = "v2.0.0/lists/edf_01_tcp_ar.list"
else:
    list_path = "v2.0.0/lists/edf_02_tcp_le.list"

# read the list of all edf file names
with open(f"{args.input_path}/{list_path}", "r") as f:
    # copy all files into one folder
    for file in tqdm(f):
        path = file.replace("..", "")[:-1]
        path = f"{args.input_path}/v2.0.0{path}"
        shutil.copy(path, args.export_path)
