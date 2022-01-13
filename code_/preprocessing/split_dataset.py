import numpy as np
import pandas as pd
import dask
import dask.array as da
import tensorflow as tf

dask.config.set({"array.slicing.split_large_chunks": False})


class Split:

    def __init__(self, chunks, labels, patientID):
        """
        Initialization of the Split object.

        Args:
        chunks (np.array):      Array of chunks of EEG data
        labels (np.array):      Array of labels corresponding to the chunks
        patientID (np.array):   Array of patient IDs corresponding to the chunks
        """

        self.chunks = chunks
        self.labels = labels
        self.patientID = patientID

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
        print(f"        Test dataset: {round(len(test_dataset_chunks) / len(list_patientID), 4)}")

        train_labels = pd.DataFrame(train_dataset_labels).groupby(0).size()
        val_labels = pd.DataFrame(val_dataset_labels).groupby(0).size()
        test_labels = pd.DataFrame(test_dataset_labels).groupby(0).size()
        print(train_labels[1] / train_labels.sum(), "\n", val_labels[1] / val_labels.sum(), "\n", test_labels[1] / test_labels.sum())

        print("    Split the data...")
        # split in train, validation and test dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_chunks.compute().astype("float32"), train_dataset_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_dataset_chunks.compute().astype("float32"), val_dataset_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_chunks.compute().astype("float32"), test_dataset_labels))
        print("    Splitting completed")

        return train_dataset, val_dataset, test_dataset
