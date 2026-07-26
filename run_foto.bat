@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

pushd "%~dp0"

if "%~1"=="" (
    "C:\Users\vyach\AppData\Local\Python\pythoncore-3.14-64\python.exe" auto_find.py
) else (
    "C:\Users\vyach\AppData\Local\Python\pythoncore-3.14-64\python.exe" auto_find.py %*
)

popd
pause