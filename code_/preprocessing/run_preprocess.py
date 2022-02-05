import argparse
import tensorflow as tf
import dask
import os
import dask.array as da
from code_.preprocessing.preprocess import Preprocessing
from code_.preprocessing.split_dataset import Split

parser = argparse.ArgumentParser(description="Chunking, labeling and splitting the dataset")
parser.add_argument("input_path", help="Path to the root folder")
parser.add_argument("export_path", help="Path where the output should be exported")
parser.add_argument("-model", type=str, help="Name of the neural network model")
parser.add_argument("-std", type=str, choices=("True", "False"), help="Standardize each chunk to mean = 0 and standard deviation = 1")
parser.add_argument("-sec", type=int, help="Amount of seconds for each chunk")
parser.add_argument("-overlap", type=str, choices=("True", "False"), help="Defines whether the chunks should overlap or not")
parser.add_argument("-exclude", type=float, help="Coefficient to calculate the threshold of exclusion criterion")
parser.add_argument("-num", type=int, help="Number of identical values allowed to occur after each other")
args = parser.parse_args()
std = args.std == "True"
overlap = args.overlap == "True"

dask.config.set({"array.slicing.split_large_chunks": False})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# initialize Preprocessing object
preprocessor = Preprocessing(args.model, std, args.sec, overlap, args.exclude, args.num)

# modify the tcp_ar labels
df_labels_ar = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_01_tcp_ar.csv")
# modify the tcp_le labels
df_labels_le = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_02_tcp_le.csv")

# go through all files and preprocess them
tuples = zip(["data/files_processed_ar", "data/files_processed_le"], [df_labels_ar, df_labels_le])
list_chunks, list_labels, list_patientID = preprocessor.preprocess_all_files(tuples)

# save list_chunks as a dask array to save memory
if args.model == "lstm":
    list_chunks = da.from_array(list_chunks, chunks=(list_chunks.shape[0]//4, list_chunks.shape[1]//4, list_chunks.shape[2]//4))
elif args.model == "hist_cnn":
    list_chunks = da.from_array(list_chunks, chunks=(
        list_chunks.shape[0] // 4, list_chunks.shape[1] // 4, list_chunks.shape[2] // 4, list_chunks.shape[3]))

print("Split the data into train-, validation- and test dataset")
splitter = Split(list_chunks, list_labels, list_patientID)
train_dataset, val_dataset, test_dataset = splitter.split_dataset()

# save train-, val- and test dataset in corresponding directory
print("Save train-, validation and test dataset")
tf.data.experimental.save(train_dataset, f"{args.export_path}/train", compression="GZIP")
tf.data.experimental.save(val_dataset, f"{args.export_path}/val", compression="GZIP")
tf.data.experimental.save(test_dataset, f"{args.export_path}/test", compression="GZIP")
print("Saved")

