@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -m streamlit run app.py
  if %errorlevel%==0 goto :eof
  py -3.10 -m streamlit run app.py
  if %errorlevel%==0 goto :eof
  py -3 -m streamlit run app.py
  if %errorlevel%==0 goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
  python -m streamlit run app.py
  if %errorlevel%==0 goto :eof
)

echo Failed to launch Streamlit.
echo Please install Python 3.10 or 3.11 and Streamlit (pip install -r requirements.txt).
pause
