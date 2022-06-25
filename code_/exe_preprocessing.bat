@ECHO OFF

SET res=F
IF %prep_type%=="normal" SET res=T
IF %prep_type%=="all" SET res=T
IF %res%==T (
    ECHO Chunk, label and split the data
    REM preprocessing data for LSTM model
    python -m code_.preprocessing.run_preprocess %root% "data/prep_normal/T_2_T_002_10" -prep %prep_type% -std "True" -sec 2 -overlap "True" -exclude 0.02 -num 10
)

SET res=F
IF %prep_type%=="contour" SET res=T
IF %prep_type%=="all" SET res=T
IF %res%==T (
    ECHO Chunk, label and split the data for the Histogram CNN model
    REM preprocessing data for Hist-CNN model
    python -m code_.preprocessing.run_preprocess %root% "data/prep_contour/T_10_F_002_10" -prep %prep_type% -std "True" -sec 10 -overlap "False" -exclude 0.02 -num 10
)

REM IMPLEMENT THE CASE THAT SOMEONE TYPES IN SOMETHING STUPID AS A MODEL TYPE