@echo off
chcp 65001 > NUL
set "PYTHONUTF8=1"
set "PYTHON=C:\Users\vyach\AppData\Local\Python\pythoncore-3.14-64\python.exe"
pushd "%~dp0"
if "%~1"=="" (
    "%PYTHON%" "process_foto.py"
) else (
    "%PYTHON%" "process_foto.py" %*
)
popd
pause
