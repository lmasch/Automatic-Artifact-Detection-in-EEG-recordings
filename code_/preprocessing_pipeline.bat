@ECHO OFF

REM SET root="D:/EEG dataset"
SET root=%1
SET model=%2

ECHO Run nedc_pystream.py
REM CALL code_/exe_nedc_pystream.bat

ECHO Run peprocessing for respective models
CALL code_/exe_preprocessing.bat

