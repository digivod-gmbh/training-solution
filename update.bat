@echo off

REM go to directory of update.bat (needed for admin mode)
cd %~dp0

git status
echo.
echo.

setlocal
:PROMPT
SET /P AREYOUSURE=Updating will reset all local changes. Do you want to continue (Y/[N])?
IF /I "%AREYOUSURE%" NEQ "Y" GOTO END

git checkout -- .
git pull

REM update TrainingSolution
pip install -e .

REM setx QT_AUTO_SCREEN_SCALE_FACTOR "1"
setx MXNET_GPU_MEM_POOL_TYPE "Naive"
setx MXNET_CUDNN_AUTOTUNE_DEFAULT "0"
setx MXNET_HOME "%~dp0%networks\\"

:END
endlocal

pause
