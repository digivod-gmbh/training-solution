@echo off

REM go to directory of install.bat (needed for admin mode)
cd %~dp0

pip install numpy==1.14.6
pip install Cython==0.29.13
pip install -e ./dependencies/pycocotools/PythonAPI/

REM install TrainingSolution
pip install -e .

pause