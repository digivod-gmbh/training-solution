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

:END
endlocal

pause
