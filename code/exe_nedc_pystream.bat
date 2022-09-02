@ECHO OFF

REM execute copy_files.py
ECHO Copying edf files from "01_tcp_ar" to folder "files_ar"
python -m code.preprocessing.copy_files %root% "data/files_ar" -ar

ECHO Copying edf files from "02_tcp_le" to folder "files_le"
python -m code.preprocessing.copy_files %root% "data/files_le" -le

REM execute nedc_pystream.py for the 01_tcp_ar data
ECHO Preprocessing with nedc_pystream for the 01_tcp_ar data. This may take a while...
for /f "delims==" %%a IN ('dir /b /s "data/files_ar"') do (
    call python -m code.preprocessing.nedc_pystream -p params_04.txt -export "data/files_processed_ar" "%%a"
    echo %%a processed successfully
)
ECHO Completed

REM execute nedc_pystream.py for the 02_tcp_le data
ECHO Preprocessing with nedc_pystream for the 02_tcp_le data. This may take a while...
for /f "delims==" %%a IN ('dir /b /s "data/files_le"') do (
    call python -m code.preprocessing.nedc_pystream -p params_05.txt -export "data/files_processed_le" "%%a"
    echo %%a processed successfully
)
ECHO Completed
