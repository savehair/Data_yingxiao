@echo off
set PY=E:\JetBrains\Anaconda3\envs\pytorch\python.exe
%PY% -c "import sys; print(sys.executable)"
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\01_build_stagewise_composite_factors.py
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\02_forward_stagewise_selection.py --window-size 12 --max-features 15
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\03_promethee_stagewise.py --top-n-promethee-features 20
if errorlevel 1 exit /b %errorlevel%
%PY% scripts\04_backtest_rating_stagewise.py
if errorlevel 1 exit /b %errorlevel%
