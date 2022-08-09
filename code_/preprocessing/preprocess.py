import os
import dask
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import dask.array as da
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
matplotlib.use('Agg')
dask.config.set({"array.slicing.split_large_chunks": False})


class Preprocessing:

    def __init__(self, prep_type, std, sec, overlap, exclude, num, image_size):
        """
        Initialization of the Preprocessing object

        Args:
        prep_type (String):     Indicates the preprocessing type (either "normal" or "contour").
        std (bool):             Standardize each chunk, current and previous chunk or entire recording
                                to mean = 0 and standard deviation = 1.
        sec (int):              Amount of seconds for each chunk.
        overlap (bool):         Defines whether the chunks should overlap or not.
        exclude (float):        Coefficient to calculate the threshold of exclusion criterion
        num (int):              Maximum number of constant consecutively occurring values allowed.
        image_size (int):       Size of the contour images.
        """

        self.SAMPLING_RATE = 250
        self.channels = 22
        self.include_rare = False
        self.rare_art = ["eyem_shiv", "shiv"]
        self.prep_type = prep_type
        self.std = std
        self.sec = sec
        self.overlap = overlap
        self.exclude = exclude
        self.num = num
        self.image_size = image_size

    def forward_shift(self, df, num):
        """
        Checks for each value in the DataFrame whether the next "num" values are the same value as the current one.
        Implemented recursively.

        Args:
        df (DataFrame):         DataFrame with the samples for each channel
        num (int):              Number of consecutively occurring values.

        return:
        DataFrame:              DataFrame containing boolean values.
        """

        if num == 1:
            return df.eq(df.shift(-num))

        return df.eq(df.shift(-num)) & self.forward_shift(df, num-1)

    def backward_shift(self, df, num):
        """
        Checks for each value in the DataFrame whether the previous "num" values are the same value as the current one.
        Implemented recursively.

        Args:
        df (DataFrame):         DataFrame with the samples for each channel
        num (int):              Number of consecutively occurring values.

        return:
        DataFrame:              DataFrame containing boolean values.
        """

        if num == 1:
            return df.eq(df.shift(num))

        return df.eq(df.shift(num)) & self.backward_shift(df, num-1)

    def in_between(self, df, num1, num2=0):
        """
        Checks for each value in the DataFrame whether the current value is within the consecutively occurring value
        range (of size "num"). Every possible combination is checked. Implemented recursively.

        Args:
        df (DataFrame):         DataFrame with the samples for each channel
        num1 (int):             Number of consecutively occurring values in backward direction.
        num2 (int):             Number of consecutively occurring values in forward direction.

        return:
        DataFrame:              DataFrame containing boolean values.
        """

        if num1 == 1:
            return self.backward_shift(df, num1) & self.forward_shift(df, num2)

        return (self.backward_shift(df, num1-1) & self.forward_shift(df, num2+1)) | self.in_between(df, num1-1, num2+1)

    def histogram_matrix(self, chunk):
        """
        Creates histogram matrix by computing the histogram for each channel in the given chunk.

        Args:
        chunk (np.array):       Array of the EEG data chunk

        return:
        matrix (np.array):      Array with histograms, each row represents the histogram of one channel
        """

        matrix = np.zeros((chunk.shape[1], chunk.shape[1]))
        for i in range(chunk.shape[1]):
            matrix[i, :], _ = np.histogram(chunk[:, i], bins=chunk.shape[1])

        return matrix

    def contour_plot(self, array):
        """
        Creates a contour plot from the histogram matrix and saves the plot as an RGB image.

        Args:
        array (np.array):           Array containing the histogram matrix.

        return:
        contour_plot (np.array):    3D array, contour plot image
        """

        hist_matrix = self.histogram_matrix(array)

        fig, ax = plt.subplots(figsize=(self.image_size, self.image_size))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis("off")
        fig.add_axes(ax)
        plt.contourf(hist_matrix, levels=self.channels, cmap="jet")
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        contour_plot = data.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return np.asarray(contour_plot)

    def chunk_label(self, df, df_labels, identifier):
        """
        This function preprocesses the TUH EEG data dependent on the specific model indicated for usage.

        Args:
        df (DataFrame):         DataFrame containing the channels with respective samples.
        df_labels (DataFrame):  DataFrame containing the start and stop time points for artifacts in each channel.
        ID (int):               ID of the patient.

        return:
        chunks (np.array):      Chunked EEG data.
        labels (np.array):      Labels of the EEG data chunks.
        """

        # filter out consecutively occurring values
        df_cond = self.forward_shift(df, self.num) | self.backward_shift(df, self.num) | self.in_between(df, self.num)
        df = df[df_cond == False].dropna(thresh=16).fillna(df)
        # Standardization of entire recording
        df = (df - df.mean(axis=0)) / df.std(axis=0) if self.std == "Entire" else df

        additional = 1
        end = 0
        # if overlap is False and the amount of seconds is bigger than 1 we need to
        # adjust the step size and the end parameter
        if self.overlap == False:
            end = len(df.iloc[:, 0]) % (self.SAMPLING_RATE * self.sec)
            additional = self.sec
        elif self.overlap:
            end = len(df.iloc[:, 0]) % self.SAMPLING_RATE + self.SAMPLING_RATE

        # create lists which store the data chunks and the labels
        chunks = []
        labels = []
        # for each specified amount of seconds (sec) cut out the chunks (dependent on overlap)
        # start from 0, end
        for i in range(0, len(df.iloc[:, 0]) - end, self.SAMPLING_RATE * additional):
            # cut out the specified amount of seconds
            df_tmp = df.iloc[i:i + self.SAMPLING_RATE * self.sec, :]

            if self.std == "True":
                # standardize the data on current chunk
                df_tmp = (df_tmp - df_tmp.mean(axis=0)) / df_tmp.std(axis=0)
            elif self.std =="PrevCur":
                # standardizes the current- + previous chunk
                if i - self.SAMPLING_RATE * self.sec < 0:
                    df_dub = df.iloc[:i + self.SAMPLING_RATE * self.sec * 2, :]
                    # standardize
                    df_dub = (df_dub - df_dub.mean(axis=0)) / df_dub.std(axis=0)
                    df_tmp = df_dub.iloc[i:i + self.SAMPLING_RATE * self.sec, :]
                else:
                    df_dub = df.iloc[i - self.SAMPLING_RATE * self.sec:i + self.SAMPLING_RATE * self.sec, :]
                    # standardize
                    df_dub = (df_dub - df_dub.mean(axis=0)) / df_dub.std(axis=0)
                    df_tmp = df_dub.iloc[self.SAMPLING_RATE * self.sec:, :]

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
            art = np.unique(filtered["artifact_label"])

            if self.include_rare:
                # if all artifacts in that chunk occur less than "exclude" seconds and there
                # is actually something in the filtered data frame we exclude this chunk
                if not filtered.empty and (np.all(np.all(filtered["duration"] < self.exclude) or (np.all(np.isin(art, self.rare_art))))):
                    continue
            else:
                # if all artifacts in that chunk occur less than "exclude" seconds and there
                # is actually something in the filtered data frame we exclude this chunk
                if not filtered.empty and np.all(filtered["duration"] < self.exclude):
                    continue

            # extra preprocessing steps for the CNN with contour plots
            df_tmp = self.contour_plot(np.asarray(df_tmp)) if self.prep_type == "contour" else df_tmp

            # append the dataframe to the chunk list
            chunks.append(df_tmp)
            # append boolean value of whether the current chunk contains artifact or not
            labels.append(not selected_l.empty)

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
        # save list_chunks as a dask array to save memory
        if self.prep_type == "normal":
            list_chunks = da.empty((0, self.SAMPLING_RATE * self.sec, self.channels),
                                   chunks=(100, self.SAMPLING_RATE * self.sec, self.channels),
                                   dtype="float32")
        elif self.prep_type == "contour":
            list_chunks = da.empty((0, self.image_size * 100, self.image_size * 100, 3),
                                   chunks=(100, self.image_size * 100, self.image_size * 100, 3),
                                   dtype="float32")

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

                    list_chunks = da.append(list_chunks, chunks, axis=0)
                    list_labels.extend(labels)
                    list_patientID.extend(np.tile(patientID, labels.shape[0]))

        return list_chunks, np.asarray(list_labels), np.asarray(list_patientID)

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
        df_labels = df_labels.rename(mapper={'# key': 'key',  "channel_label": "channel_label",
                                             " start_time": "start_time", " stop_time": "stop_time",
                                             " artifact_label": "artifact_label"}, axis='columns')

        return df_labels

