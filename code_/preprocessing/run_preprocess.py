import argparse
import tensorflow as tf
import numpy as np
import dask
import dask.array as da
from code_.preprocessing.preprocess import Preprocessing
from code_.preprocessing.split_dataset import Split

dask.config.set({"array.slicing.split_large_chunks": False})

parser = argparse.ArgumentParser(description="Chunking, labeling and splitting the dataset")
parser.add_argument("input_path", help="Path to the root folder")
parser.add_argument("export_path", help="Path where the output should be exported")
parser.add_argument("-std", type=bool, help="Standardize the data to mean = 0 and standard deviation = 1")
parser.add_argument("-sec", type=int, help="Amount of seconds for each chunk")
parser.add_argument("-overlap", type=bool, help="Defines whether the chunks should overlap or not")
parser.add_argument("-exclude", type=float, help="Coefficient to calculate the threshold of exclusion criterion")
args = parser.parse_args()

# initialize Preprocessing object
preprocessor = Preprocessing(args.std, args.sec, args.overlap, args.exclude)

# modify the tcp_ar labels
df_labels_ar = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_01_tcp_ar.csv")
# modify the tcp_le labels
df_labels_le = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_02_tcp_le.csv")

# go through all files and preprocess them
tuples = zip(["data/files_processed_ar", "data/files_processed_le"], [df_labels_ar, df_labels_le])
list_chunks, list_labels, list_patientID = preprocessor.preprocess_all_files(tuples)
# save list_chunks as a dask array to save memory
list_chunks = da.from_array(list_chunks, chunks=(list_chunks.shape[0]//4, list_chunks.shape[1]//4, list_chunks.shape[2]//4))

print("Split the data into train-, validation- and test dataset")
splitter = Split(list_chunks, list_labels, list_patientID)
train_dataset, val_dataset, test_dataset = splitter.split_dataset()

# save train-, val- and test dataset in corresponding directory
print("Save train-, validation and test dataset")
tf.data.experimental.save(train_dataset, "data/tensorflow_datasets_lstm/train", compression="GZIP")
tf.data.experimental.save(val_dataset, "data/tensorflow_datasets_lstm/val", compression="GZIP")
tf.data.experimental.save(test_dataset, "data/tensorflow_datasets_lstm/test", compression="GZIP")
print("Saved")

