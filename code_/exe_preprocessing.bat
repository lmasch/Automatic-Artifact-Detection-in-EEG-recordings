@ECHO OFF

ECHO Chunk, label and split the data
REM preprocess data
python -m code_.preprocessing.run_preprocess %root% "data/LSTM/preprocessed_files" -std True -sec 2 -overlap True -exclude 0.2
