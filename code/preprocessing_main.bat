@ECHO OFF

SET root=%1
REM "data/prep_normal/T_2_T_002_10_entireRecording" or "data/prep_contour/T_30_F_002_10"
SET export_directory=%2
SET prep_type=%3
SET std=%4
SET sec=%5
SET overlap=%6
SET exclude=%7
SET num=%8

CALL code/exe_preprocessing.bat

