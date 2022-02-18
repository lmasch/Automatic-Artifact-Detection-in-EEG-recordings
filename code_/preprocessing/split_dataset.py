import numpy as np
import pandas as pd
import dask
import os
import dask.array as da
dask.config.set({"array.slicing.split_large_chunks": False})


class Split:

    def __init__(self, chunks, labels, patientID, export_path, sec, model, batch_size, image_size):
        """
        Initialization of the Split object.

        Args:
        chunks (np.array):      Array of chunks of EEG data
        labels (np.array):      Array of labels corresponding to the chunks
        patientID (np.array):   Array of patient IDs corresponding to the chunks
        """

        self.SAMPLING_RATE = 250
        self.channels = 22
        self.sec = sec
        self.chunks = chunks
        self.labels = labels
        self.patientID = patientID
        self.export_path = export_path
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size

    def find_nearest(self, array, value):
        """
        This function finds the index of the number within the dataset which is closest to "value".

        Args:
        array (array):    Array with float values.
        value (float):    Float value.

        return:
        idx (int):        Index of the array where closest value resides.
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx

    def balance_data(self):
        """
        This function balances our data such that it consists of 50% artifacts and 50% without artifacts

        return:
        list_chunks (np.array):         Array of chunks of EEG data
        list_labels (np.array):         Array of labels
        list_patientID (np.array):      Array of patient IDs
        """

        # tackle unbalanced data
        # find out what part of the dataset is represented more
        bool_artifact = (len(np.argwhere(self.labels)) > len(np.argwhere(self.labels == False)))
        size = len(np.argwhere(self.labels == bool_artifact)) - len(np.argwhere(self.labels != bool_artifact))
        random_idx = np.random.choice(np.argwhere(self.labels == bool_artifact).reshape(-1), size=size, replace=False)
        remaining_idx = np.delete(np.arange(len(self.labels)), random_idx)

        return self.chunks[remaining_idx], self.labels[remaining_idx], self.patientID[remaining_idx]

    def split_dataset(self):
        """
        This function splits the data into train-, validation- and test dataset.

        return:
        train_dataset (tf.data.Dataset):    train dataset, containing the chunks and labels for training
        val_dataset (tf.data.Dataset):      validation dataset, containing the chunks and labels for validation
        test_dataset (tf.data.Dataset):     test dataset, containing the chunks and labels for testing
        """

        print("    Balance the data")
        list_chunks, list_labels, list_patientID = self.balance_data()
        print("    Balancing completed", "\n")

        # get the size of the respective datasets with respect to the full dataset
        train_size = np.round(len(list_chunks) * 0.70)
        val_size = np.round(len(list_chunks) * 0.15)
        test_size = np.round(len(list_chunks) * 0.15)

        print("    Assign patient IDs randomly to one of the datasets")
        # convert patientID to a DataFrame
        IDs = pd.DataFrame(list_patientID)
        # get the unique patient IDs and count the amount of chunks
        unique_ID = IDs.groupby(0).size()
        # shuffle the patient IDs
        unique_ID = unique_ID.sample(frac=1)

        # get the index of the cumulated sum which is closest to our training size value
        idx_train = self.find_nearest(unique_ID.cumsum(), train_size)
        # extract the IDs for our training dataset
        train_IDs = unique_ID[:idx_train + 1].index

        # update the unique IDs
        unique_ID = unique_ID[idx_train + 1:]
        idx_val = self.find_nearest(unique_ID.cumsum(), val_size)
        val_IDs = unique_ID[:idx_val + 1].index

        unique_ID = unique_ID[idx_val + 1:]
        test_IDs = unique_ID.index

        # assign those chunks to the training dataset which belong to the train IDs
        train_dataset_chunks = list_chunks[np.argwhere(np.isin(list_patientID, train_IDs)).reshape(-1)]
        train_dataset_labels = list_labels[np.argwhere(np.isin(list_patientID, train_IDs)).reshape(-1)]
        # assign those chunks to the validation dataset which belong to the validation IDs
        val_dataset_chunks = list_chunks[np.argwhere(np.isin(list_patientID, val_IDs)).reshape(-1)]
        val_dataset_labels = list_labels[np.argwhere(np.isin(list_patientID, val_IDs)).reshape(-1)]
        # assign those chunks to the test dataset which belong to the test IDs
        test_dataset_chunks = list_chunks[np.argwhere(np.isin(list_patientID, test_IDs)).reshape(-1)]
        test_dataset_labels = list_labels[np.argwhere(np.isin(list_patientID, test_IDs)).reshape(-1)]

        print("    Assigning completed", "\n")
        print("        Dataset proportions:")
        print(f"        Train dataset: {round(len(train_dataset_chunks) / len(list_patientID), 4)}")
        print(f"        Validation dataset: {round(len(val_dataset_chunks) / len(list_patientID), 4)}")
        print(f"        Test dataset: {round(len(test_dataset_chunks) / len(list_patientID), 4)}, \n")

        train_labels = pd.DataFrame(train_dataset_labels).groupby(0).size()
        val_labels = pd.DataFrame(val_dataset_labels).groupby(0).size()
        test_labels = pd.DataFrame(test_dataset_labels).groupby(0).size()

        print(f"        Label ratio:")
        print(f"        Train dataset: {train_labels[1] / train_labels.sum()}")
        print(f"        Val dataset: {val_labels[1] / val_labels.sum()}")
        print(f"        Test dataset: {test_labels[1] / test_labels.sum()}")

        print("    Split and save the data...")
        # remove previously exported hdf5 files
        for filename in os.listdir(self.export_path):
            f = os.path.join(self.export_path, filename)
            # checking if it is a file
            if f.endswith(".hdf5"):
                os.remove(f)

        # define chunk shapes for respective models
        if self.model == "lstm":
            chunk_shape = (self.batch_size, self.SAMPLING_RATE * self.sec, self.channels)
        elif self.model == "hist_cnn":
            chunk_shape = (self.batch_size, self.image_size * 100, self.image_size * 100, 3)
        elif self.model == "cnn":
            chunk_shape = (self.batch_size, self.SAMPLING_RATE * self.sec, self.channels)

        # split in train, validation and test dataset and save it as hdf5 files
        da.to_hdf5(f"{self.export_path}/train_dataset.hdf5",
                   {"/samples": da.rechunk(train_dataset_chunks, chunks=chunk_shape),
                    "/labels": da.from_array(train_dataset_labels)},
                   compression="gzip")
        da.to_hdf5(f"{self.export_path}/val_dataset.hdf5",
                   {"/samples": da.rechunk(val_dataset_chunks, chunks=chunk_shape),
                    "/labels": da.from_array(val_dataset_labels)},
                   compression="gzip")
        da.to_hdf5(f"{self.export_path}/test_dataset.hdf5",
                   {"/samples": da.rechunk(test_dataset_chunks, chunks=chunk_shape),
                    "/labels": da.from_array(test_dataset_labels)},
                   compression="gzip")

        print("    Saved")
