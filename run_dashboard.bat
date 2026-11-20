@echo off
cd /d "%~dp0"
pip install -r requirements.txt
streamlit run dpf_day5_dashboard.py
pause
