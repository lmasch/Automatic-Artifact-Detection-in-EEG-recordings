@ECHO OFF

ECHO Chunk, label and split the data for the LSTM model
REM preprocessing data for LSTM model
REM python -m code_.preprocessing.run_preprocess %root% "data/tensorflow_datasets_lstm" -model "lstm" -std True -sec 2 -overlap True -exclude 0.2

ECHO Chunk, label and split the data for the Histogram CNN model
REM preprocessing data for Hist-CNN model
python -m code_.preprocessing.run_preprocess %root% "data/tensorflow_datasets_hist_cnn" -model "hist_cnn" -std "True" -sec 30 -overlap "False" -exclude 0.05 -num 10