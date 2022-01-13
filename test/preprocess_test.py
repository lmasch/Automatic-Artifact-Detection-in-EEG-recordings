import unittest
import numpy as np
import pandas as pd
import dask.array as da
from code_.preprocessing.preprocess import Preprocessing
from code_.preprocessing.split_dataset import Split


class PreprocessTest(unittest.TestCase):

    def setUp(self):
        self.SAMPLING_RATE = 250

        # create 100 chunks
        cols = np.arange(0, 22)
        samples = 250 * 100 + 250 * 5
        idx = np.linspace(0, samples / 250, samples, endpoint=False)

        self.test_df = pd.DataFrame(columns=cols, index=idx)
        self.test_df.iloc[:, :] = 1.0

        # create test_patientID
        self.test_patientID = "0"

        # create test_df_labels
        self.test_df_labels = pd.DataFrame(columns=["key", "start_time", "stop_time"])
        self.test_df_labels["key"] = np.repeat(self.test_patientID, 100)

        # create some start and stop times within test_df_labels
        # [..]
        self.test_df_labels.iloc[0:10, 1] = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        self.test_df_labels.iloc[0:10, 2] = np.array([1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9])
        # .[.]
        self.test_df_labels.iloc[10:20, 1] = np.array([19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5])
        self.test_df_labels.iloc[10:20, 2] = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        # [.].
        self.test_df_labels.iloc[20:30, 1] = np.array([40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
        self.test_df_labels.iloc[20:30, 2] = np.array([42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5])
        # .[].
        self.test_df_labels.iloc[30:40, 1] = np.array([59, 60, 61, 62, 63, 64, 65, 66, 67, 68])
        self.test_df_labels.iloc[30:40, 2] = np.array([63, 64, 65, 66, 67, 68, 69, 70, 71, 72.08])
        # too short duration
        self.test_df_labels.iloc[40:43, 1] = np.array([80.96, 82, 83])
        self.test_df_labels.iloc[40:43, 2] = np.array([82, 82.09, 83.001])

        self.test_df_labels = self.test_df_labels.dropna()

    def test_label_correctness(self):

        preprocessor = Preprocessing(False, 2, True, 0.05)
        chunks, labels = preprocessor.chunk_label(self.test_df, self.test_df_labels, self.test_patientID)
        # create the true labels
        true_labels = np.zeros(100)
        true_labels[:11] = np.ones(11)
        true_labels[18:30] = np.ones(12)
        true_labels[39:52] = np.ones(13)
        true_labels[58:72] = np.ones(14)
        true_labels[78:80] = 1
        true_labels = true_labels.astype("bool")

        self.assertSequenceEqual(true_labels.tolist(), labels.tolist())

    def test_len_chunks(self):

        preprocessor = Preprocessing(False, 2, True, 0)
        chunks, labels = preprocessor.chunk_label(self.test_df, self.test_df_labels, self.test_patientID)
        sec = len(self.test_df)/self.SAMPLING_RATE
        len_chunks = sec-1

        # test if the length of the chunks is the same
        self.assertEqual(len_chunks, len(chunks), "No correct amount of chunks extracted from the file.")

    def test_splitting(self):

        preprocessor = Preprocessing(False, 2, True, 0.05)
        patientIDs = [str(i) for i in range(9)]

        df_labels = pd.DataFrame(columns=["key", "start_time", "stop_time"])
        df_labels["key"] = np.repeat(patientIDs, 10)

        df_labels.iloc[0:10, 1] = 3.5
        df_labels.iloc[0:10, 2] = 4

        df_labels.iloc[10:20, 1] = 12.5
        df_labels.iloc[10:20, 2] = 13

        df_labels.iloc[20:30, 1] = 24.5
        df_labels.iloc[20:30, 2] = 25

        df_labels.iloc[30:40, 1] = 33.5
        df_labels.iloc[30:40, 2] = 34

        df_labels.iloc[40:50, 1] = 45.5
        df_labels.iloc[40:50, 2] = 46

        df_labels.iloc[50:60, 1] = 51.5
        df_labels.iloc[50:60, 2] = 52

        df_labels.iloc[60:70, 1] = 64.5
        df_labels.iloc[60:70, 2] = 65

        df_labels.iloc[70:80, 1] = 77.5
        df_labels.iloc[70:80, 2] = 78

        df_labels.iloc[80:90, 1] = 84.5
        df_labels.iloc[80:90, 2] = 85

        tuples = zip(["test/test_data"], [df_labels])
        list_chunks, list_labels, list_patientID = preprocessor.preprocess_all_files(tuples)
        list_chunks = da.from_array(list_chunks, chunks=(list_chunks.shape[0] // 4, list_chunks.shape[1] // 4, list_chunks.shape[2] // 4))

        splitter = Split(list_chunks, list_labels, list_patientID)
        train_dataset, val_dataset, test_dataset = splitter.split_dataset()

        samples = []
        labels = []
        for sample, label in train_dataset:
            samples.append(sample)
            labels.append(label)

        for sample, label in val_dataset:
            samples.append(sample)
            labels.append(label)

        for sample, label in test_dataset:
            samples.append(sample)
            labels.append(label)

        samples = np.asarray(samples)
        labels = np.asarray(labels)

        self.assertGreater(np.all(np.unique(samples[np.argwhere(labels)])), 0)
        self.assertEqual(np.all(np.unique(samples[np.argwhere(labels == False)])), 0)


if __name__ == "__main__":
    unittest.main()