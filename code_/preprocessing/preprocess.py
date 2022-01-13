import os
from tqdm import tqdm
import numpy as np
import pandas as pd


class Preprocessing:

    def __init__(self, std=True, sec=2, overlap=True, exclude=0.05):
        """
        Initialization of the Preprocessing object

        Args:
        std (bool):             Standardize the data to mean = 0 and standard deviation = 1.
        sec (int):              Amount of seconds for each chunk.
        overlap (bool):         Defines whether the chunks should overlap or not.
        exclude (float):        Coefficient to calculate the threshold of exclusion criterion
        """

        self.SAMPLING_RATE = 250
        self.std = std
        self.sec = sec
        self.overlap = overlap
        self.exclude = exclude

    def chunk_label(self, df, df_labels, identifier):
        """
        This function preprocesses the TUH EEG data dependent on the specific model to use.

        Args:
        df (DataFrame):         DataFrame containing the channels with respective samples.
        df_labels (DataFrame):  DataFrame containing the start and stop time points for artifacts in each channel.
        ID (int):               ID of the patient.

        return:
        chunks (np.array):      Chunked EEG data.
        labels (np.array):      Labels of the EEG data chunks.
        """

        # copy the DataFrame
        df_norm = df.copy()

        # standardize the data
        if self.std:
            df_norm = (df_norm - df_norm.mean(axis=0)) / df_norm.std(axis=0)

        additional = 1
        end = 0
        # if overlap is False and the amount of seconds is bigger than 1 we need to
        # adjust the step size in the for loop
        if self.overlap == False and self.sec > 1:
            additional = self.sec
        # if overlap is true we have to adjust the end parameter for our for-loop
        elif self.overlap:
            end = self.SAMPLING_RATE

        # create lists which store the data chunks and the labels
        chunks = []
        labels = []
        # for each specified amount of seconds (sec) cut out the chunks (dependent on overlap)
        # start from 0, end
        for i in range(0, len(df_norm.iloc[:, 0]) - end, self.SAMPLING_RATE * additional):
            # cut out the specified amount of seconds
            df_tmp = df_norm.iloc[i:i + self.SAMPLING_RATE * self.sec, :]

            # search for the rows with ID's in question
            df_ID = df_labels[df_labels["key"] == identifier]
            # select those rows from the artifact data which occur in the current chunk
            selected_l = df_ID[((df_ID["start_time"] < df_tmp.index[0]) & (df_ID["stop_time"] > df_tmp.index[-1]))
                               | ((df_ID["start_time"] > df_tmp.index[0]) & (df_ID["stop_time"] < df_tmp.index[-1]))
                               | ((df_ID["start_time"] > df_tmp.index[0]) & (df_ID["start_time"] < df_tmp.index[-1]))
                               | ((df_ID["stop_time"] > df_tmp.index[0]) & (df_ID["stop_time"] < df_tmp.index[-1]))]

            # extract the duration of the artifacts in the segment
            filtered = selected_l.copy()
            filtered["start"] = filtered["start_time"].apply(lambda x: df_tmp.index[0] if x < df_tmp.index[0] else x)
            filtered["stop"] = filtered["stop_time"].apply(lambda x: df_tmp.index[-1] if x > df_tmp.index[-1] else x)
            filtered["duration"] = filtered["stop"] - filtered["start"]

            # if all artifacts in that chunk occur less than "sec * exclude" seconds and there
            # is actually something in the filtered data frame we exclude this chunk
            if not(filtered).empty and np.all(filtered["duration"] < self.sec * self.exclude):
                continue

            # append the dataframe to the chunk list
            chunks.append(df_tmp)
            # append boolean value of whether the current chunk contains artifact or not
            labels.append(not(selected_l.empty))

        return np.asarray(chunks), np.asarray(labels)

    def preprocess_all_files(self, tuples):
        """
        This function iterates over each EEG recording and applies the "preprocessing" function.

        Args:
        tuples (zip):           Contains the paths to the nedc-preprocessed files and corresponding artifact label csv file

        return:
        list_chunks (list):     The array with all EEG data chunks
        list_labels (list):     The array with all labels for the corresponding EEG chunks
        list_patientID (list):  The array with the patient IDs of corresponding EEG chunks
        """

        # initialize lists for the chunks, labels and patient IDs
        list_chunks = []
        list_labels = []
        list_patientID = []
        for path, df_labels in tuples:

            for filename in tqdm(os.listdir(path)):
                f = os.path.join(path, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    # get the identifier of the recording
                    identifier = filename[:-4]
                    # get the patient ID of the file
                    patientID = filename[:8]

                    df = pd.read_pickle(f)
                    chunks, labels = self.chunk_label(df, df_labels, identifier)

                    list_chunks.extend(chunks)
                    list_labels.extend(labels)
                    list_patientID.extend(np.tile(patientID, labels.shape[0]))

        return np.asarray(list_chunks), np.asarray(list_labels), np.asarray(list_patientID)

    def modify_label_csv(self, path):
        """
        This function remaps the column names of the given DataFrame to other names.

        Args:
        path (string):          Path to the csv file with the labels

        return:
        df_labels (DataFrame):  Modified DataFrame
        """

        # modify labels for tcp_ar csv
        df_labels = pd.read_csv(path, skiprows=4)
        df_labels = df_labels.dropna()
        # remapping the columns
        df_labels = df_labels.rename(mapper={'# key': 'key',  "channel_label": "channel_label",  " start_time": "start_time", " stop_time": "stop_time", " artifact_label": "artifact_label"}, axis='columns')

        return df_labels

