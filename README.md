# DPF SCV – Day 5 reporting dashboard

A Streamlit dashboard for Day 5 of the DPF data training: anomaly review, rules vs model,
the awareness model, the capstone run, and **every chart from the training notebook**
(Sankeys, waterfall, treemap, map and more).

> **Dummy training data only.** Everything here was produced from `DPF_SCV_Training_Dummy_Data.xlsx`.
> Never put real SCV data in this folder if it is published online.

## Run it on your computer
1. Install Python 3.11 (Anaconda is fine).
2. In a terminal / Anaconda Prompt, in this folder:
   ```
   pip install -r requirements.txt
   streamlit run dpf_day5_dashboard.py
   ```
   Or double-click `run_dashboard.bat` (Windows) / `run_dashboard.command` (Mac).
3. A browser tab opens at http://localhost:8501. Stop it with Ctrl + C.

## What is in this folder
| File | What it is |
|---|---|
| `dpf_day5_dashboard.py` | The dashboard (commented, one function per tab) |
| `DPF_outputs/day5/` | The notebook's saved Day 5 results that the dashboard reads |
| `DPF_SCV_5Day_Training_NotebookV2.ipynb` | The training notebook; the *Notebook charts* tab shows its charts |
| `DPF_SCV_Training_Dummy_Data.xlsx` | The dummy data everything was made from |
| `requirements.txt` | Exact package versions (the saved results need these pandas / scikit-learn versions) |

## Update the results
Run the notebook from the top (section 5.7, STEP 1 and STEP 2), save it, then copy
`DPF_outputs/day5/` and the notebook into this folder.

## Deploy on Streamlit Community Cloud
- Repository: this repo · Branch: `main` · Main file path: `dpf_day5_dashboard.py`
- Advanced settings → Python version: **3.11** (versions are pinned in `requirements.txt`)
