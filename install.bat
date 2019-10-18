@echo off

REM go to directory of install.bat (needed for admin mode)
cd %~dp0

pip install numpy==1.14.6
pip install Cython==0.29.13
pip install -e ./dependencies/pycocotools/PythonAPI/

REM install TrainingSolution
pip install -e .

REM setx QT_AUTO_SCREEN_SCALE_FACTOR "1"
setx MXNET_GPU_MEM_POOL_TYPE "Naive"
setx MXNET_CUDNN_AUTOTUNE_DEFAULT "0"
setx MXNET_HOME "%~dp0%networks\\"

pause