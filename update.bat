@echo off

REM go to directory of update.bat (needed for admin mode)
cd %~dp0

git pull
call install.bat