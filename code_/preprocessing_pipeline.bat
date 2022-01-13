@ECHO OFF

SET root="D:/EEG dataset"

ECHO Run nedc_pystream.py
REM CALL code_/exe_nedc_pystream.bat

ECHO Preprocessing for the LSTM model
CALL code_/exe_preprocessing.bat



REM ECHO Preprocessing for the CNN model
