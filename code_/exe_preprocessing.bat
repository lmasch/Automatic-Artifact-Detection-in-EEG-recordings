@ECHO OFF

SET res=F
IF %model%=="lstm" SET res=T
IF %model%=="all" SET res=T
IF %res%==T (
    ECHO Chunk, label and split the data for the LSTM model
    REM preprocessing data for LSTM model
    python -m code_.preprocessing.run_preprocess %root% "data/tensorflow_datasets_lstm" -model "lstm" -std "False" -sec 2 -overlap "True" -exclude 0.2 -num 10
)

SET res=F
IF %model%=="hist_cnn" SET res=T
IF %model%=="all" SET res=T
IF %res%==T (
    ECHO Chunk, label and split the data for the Histogram CNN model
    REM preprocessing data for Hist-CNN model
    python -m code_.preprocessing.run_preprocess %root% "data/tensorflow_datasets_hist_cnn" -model "hist_cnn" -std "True" -sec 30 -overlap "False" -exclude 0.05 -num 10
)

REM IMPLEMENT THE CASE THAT SOMEONE TYPES IN SOMETHING STUPID AS A MODEL TYPE