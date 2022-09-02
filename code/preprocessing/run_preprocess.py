import argparse
import os
import pandas as pd
from code.preprocessing.preprocess import Preprocessing
from code.preprocessing.split_dataset import Split
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description="Chunking, labeling and splitting the dataset")
parser.add_argument("input_path", help="Path to the root folder")
parser.add_argument("export_path", help="Path where the output should be exported")
parser.add_argument("-prep", type=str, help="Define the preprocessing type")
parser.add_argument("-std", type=str, choices=("True", "False", "PrevCur", "Entire"), help="Standardization based on current chunk or current- & previous chunk, or entire recording")
parser.add_argument("-sec", type=int, help="Amount of seconds for each chunk")
parser.add_argument("-overlap", type=str, choices=("True", "False"), help="Defines whether the chunks should overlap or not")
parser.add_argument("-exclude", type=float, help="Coefficient to calculate the threshold of exclusion criterion")
parser.add_argument("-num", type=int, help="Number of identical values allowed to occur after each other")
args = parser.parse_args()
overlap = args.overlap == "True"

# define image size for the hist_cnn model
image_size = 2
# initialize Preprocessing object
preprocessor = Preprocessing(args.prep, args.std, args.sec, overlap, args.exclude, args.num, image_size=image_size)

# modify the tcp_ar labels
df_labels_ar = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_01_tcp_ar.csv")
# modify the tcp_le labels
df_labels_le = preprocessor.modify_label_csv(f"{args.input_path}/v2.0.0/csv/labels_02_tcp_le.csv")
# concatenate labels
df_labels = pd.concat((df_labels_ar, df_labels_le))
df_labels["key_new"] = df_labels["key"].apply(lambda x: x[:8])

# go through all files and preprocess them
tuples = zip(["data/files_processed_ar", "data/files_processed_le"], [df_labels_ar, df_labels_le])
list_chunks, list_labels, list_patientID = preprocessor.preprocess_all_files(tuples)

print("Split the data into train-, validation- and test dataset")
splitter = Split(list_chunks, list_labels, list_patientID, df_labels, args.export_path, args.sec, args.prep, batch_size=64, image_size=image_size)
splitter.split_dataset()
