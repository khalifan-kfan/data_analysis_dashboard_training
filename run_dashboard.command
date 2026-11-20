#!/bin/bash
cd "$(dirname "$0")"
pip3 install -r requirements.txt
streamlit run dpf_day5_dashboard.py
