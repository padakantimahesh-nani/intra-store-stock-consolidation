@echo off
cd /d "%~dp0"
py -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo The application could not start. Install dependencies first with:
  echo py -m pip install -r requirements.txt
  pause
)
