@ECHO OFF

SET res=F
IF %prep_type%=="normal" SET res=T
IF %prep_type%=="contour" SET res=T
IF %res%==T (
    ECHO Run the main peprocessing program
    ECHO Chunk, label and split the data
    REM create directory where the data will be saved to
    if not exist %export_directory% mkdir %export_directory%
    REM preprocess the data in the specified preprocessing mode
    python -m code.preprocessing.run_preprocess %root% %export_directory% -prep %prep_type% -std %std% -sec %sec% -overlap %overlap% -exclude %exclude% -num %num%
) ELSE ECHO This preprocessing mode is not supported.