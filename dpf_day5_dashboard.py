"""
DPF Single Customer View - Day 5 reporting dashboard
=====================================================

What it is
    A small web page that shows the results of the training notebook so a
    supervisor can read, filter and download them without opening the notebook.

Where the content comes from (everything is from the notebook)
    1. Tables and numbers: the notebook (section 5.7, "STEP 1") saves its Day 5
       results as files in DPF_outputs/day5/. This script only READS those files.
       It does no cleaning and no modelling of its own.
    2. Charts: the "Notebook charts" tab reads the saved notebook file
       (DPF_SCV_5Day_Training_NotebookV2.ipynb) and shows every chart it drew, Days 1-5,
       under the notebook's own headings. Interactive Plotly charts (Sankey, treemap,
       waterfall, map, KPI dashboard) are redrawn as interactive charts; matplotlib and
       seaborn charts are shown as the exact pictures the notebook made.
       => Save the notebook (Ctrl + S) after running it, so the saved charts are current.

How to run it
    In Anaconda Prompt, in the folder that holds this file:
        pip install streamlit "pyarrow<21"      (once)
        streamlit run dpf_day5_dashboard.py
    A browser tab opens at http://localhost:8501. Stop it with Ctrl + C.
    (Or use STEP 3 in section 5.7 of the notebook to open it inside the notebook.)

How Streamlit works, in one paragraph
    Streamlit runs this whole file from top to bottom every time the user
    touches a filter or button. Each st.something(...) call draws one piece of
    the page (a title, a chart, a table). There are no callbacks: the value of
    a filter is simply the return value of the call that draws it.
"""

import base64               # notebook pictures are stored as base64 text inside the .ipynb file
import json                 # a .ipynb notebook file is JSON
from datetime import datetime
from io import BytesIO      # an in-memory "file", used to build the Excel download
from pathlib import Path    # works with file paths on Windows and Mac alike

import joblib               # loads the trained awareness model saved by the notebook
import pandas as pd
import plotly.express as px  # interactive charts
import plotly.io as pio      # rebuilds the notebook's saved Plotly charts
import streamlit as st       # turns this script into a web page


# =============================================================================
# 1. SETTINGS
# =============================================================================

HERE = Path(__file__).parent                       # the folder this script is in

# Folder where the notebook saved the Day 5 results (STEP 1 of section 5.7).
DAY5_DIR = HERE / "DPF_outputs" / "day5"

# The training notebook; its saved charts are shown in the "Notebook charts" tab.
NOTEBOOK_FILE = HERE / "DPF_SCV_5Day_Training_NotebookV2.ipynb"

# One colour per meaning, used the same way on every chart (as in the notebook).
COLOURS = {
    "critical": "#c0392b",   # red    - blocked / flagged / a problem
    "review":   "#e6a23c",   # amber  - needs a person to look at it
    "ok":       "#2e8b57",   # green  - fine / ready
    "neutral":  "#c3c2b7",   # grey   - background, "normal"
    "dark":     "#1f3b57",   # navy   - plain bars with no good/bad meaning
}

# Colour for each payout status the notebook produces.
STATUS_COLOURS = {
    "Ready to pay":             COLOURS["ok"],
    "Ready with review":        "#8fbf6a",          # light green: ready, but check first
    "Needs payment details":    COLOURS["review"],
    "Blocked - identity":       COLOURS["critical"],
    "Blocked - amount unknown": "#7b1d14",          # dark red
}

# The columns shown in the anomaly review queue table, in this order.
REVIEW_COLUMNS = [
    "Record ID", "institution_code", "Client Type", "Account Title/Name", "Account Currency",
    "Current Balance", "Interest Due (Accrued)", "Exchange Rate Applicable",
    "anomaly_score", "anomaly_reason", "has_exception",
]

# Page title, browser-tab icon and a wide layout (use the whole screen width).
# This must be the first Streamlit call in the script.
st.set_page_config(page_title="DPF Day 5 report", page_icon="📊", layout="wide")

# Make charts and tables fill their column.
# Newer Streamlit (1.46+) spells this width="stretch"; older versions use use_container_width=True.
# WIDE holds whichever one this computer's Streamlit understands; we pass it as **WIDE.
streamlit_version = tuple(int(part) for part in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if streamlit_version >= (1, 46) else {"use_container_width": True}


# =============================================================================
# 2. LOADING THE DATA
# =============================================================================

@st.cache_data   # read the files once, then reuse them (Streamlit re-runs the script on every click)
def load_results():
    """Read every .pkl file in DAY5_DIR into a dictionary: {file name: table}.

    Returns None if the folder does not exist yet (the notebook has not been run).
    """
    if not DAY5_DIR.exists():
        return None
    return {path.stem: pd.read_pickle(path) for path in DAY5_DIR.glob("*.pkl")}


@st.cache_resource   # load the model once and keep it in memory
def load_awareness_model():
    """Load the trained awareness model: {"model": the pipeline, "features": its input columns}."""
    path = DAY5_DIR / "awareness_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_notebook_charts(notebook_path, last_saved):
    """Collect every chart saved in the notebook, in notebook order.

    A notebook file stores each cell's results ("outputs"). A matplotlib/seaborn chart
    is saved as a PNG picture ("image/png"); a Plotly chart is saved as its full
    definition ("application/vnd.plotly.v1+json"), so it can be redrawn exactly.

    `last_saved` is not used inside; it is there so the cache refreshes when the
    notebook is saved again.

    Returns a list of dicts: day, section (the heading above the chart), cell number,
    kind ("plotly" or "image") and the chart itself.
    """
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    charts = []
    day, section = "Setup", ""

    for cell_number, cell in enumerate(notebook["cells"]):
        text = "".join(cell["source"])

        # Text cells: remember the latest "# DAY n" and the latest ## / ### heading.
        if cell["cell_type"] == "markdown":
            for line in text.splitlines():
                if line.startswith("# DAY"):
                    day = line.lstrip("# ").split(" — ")[0].title()      # "# DAY 4 — Visual..." -> "Day 4"
                elif line.startswith("##"):
                    section = line.lstrip("# ").strip()
            continue

        # Code cells: keep each chart in the cell's outputs.
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "application/vnd.plotly.v1+json" in data:
                charts.append({"day": day, "section": section, "cell": cell_number,
                               "kind": "plotly", "chart": data["application/vnd.plotly.v1+json"]})
            elif "image/png" in data:
                png = data["image/png"]
                png = "".join(png) if isinstance(png, list) else png
                charts.append({"day": day, "section": section, "cell": cell_number,
                               "kind": "image", "chart": base64.b64decode(png)})
    return charts


def to_excel(sheets):
    """Turn {sheet name: table} into the bytes of one Excel workbook (for the download button)."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in sheets.items():
            # Keep the index only when it means something (e.g. the 2 x 2 table's row labels).
            keep_index = not isinstance(table.index, pd.RangeIndex)
            table.to_excel(writer, sheet_name=name[:31], index=keep_index)   # Excel allows 31 characters
    return buffer.getvalue()


def to_csv(table):
    """Turn one table into CSV bytes (for a download button)."""
    keep_index = not isinstance(table.index, pd.RangeIndex)
    return table.to_csv(index=keep_index).encode("utf-8")


def bar_chart(series, title, horizontal=False):
    """A plain one-colour bar chart of a pandas Series (labels = index, bar length = values)."""
    fig = px.bar(series, orientation="h" if horizontal else "v", title=title,
                 labels={"value": "Accounts", "index": ""}, color_discrete_sequence=[COLOURS["dark"]])
    return fig.update_layout(showlegend=False)


# ---- Read everything; stop with a clear message if the notebook has not been run ----
results = load_results()
if results is None:
    st.error(f"No Day 5 results found in `{DAY5_DIR}`. "
             "Run the notebook from the top, then the cells in section 5.7.")
    st.stop()   # nothing else on the page can work without the data

accounts = results["accounts_anomaly"]          # one row per account, with anomaly_flag / score / reason
comparison = results["model_comparison"]        # simple vs improved anomaly model (recall, precision)
metrics = results["awareness_metrics"]          # awareness model results on the test set
odds_ratios = results["awareness_odds"]         # what predicts awareness (odds ratios)
survey = results["survey"]                      # the awareness survey answers
capstone_accounts = results["capstone_accounts"]
capstone_depositors = results["capstone_depositors"]
meta = results.get("meta", pd.Series(dtype=str))   # reporting date and when the results were saved


# =============================================================================
# 3. SIDEBAR FILTERS
# =============================================================================
# The filters narrow the accounts shown in the Summary, Anomaly review and Rules vs model tabs.

st.sidebar.header("Filters")

all_banks = sorted(accounts["institution_code"].dropna().unique())
all_client_types = sorted(accounts["Client Type"].dropna().unique())

chosen_banks = st.sidebar.multiselect("Institution", all_banks, default=all_banks)
chosen_client_types = st.sidebar.multiselect("Client type", all_client_types, default=all_client_types)

st.sidebar.caption(f"Reporting date: {meta.get('reporting_date', '-')}  \n"
                   f"Results saved: {meta.get('created', '-')}")
st.sidebar.caption("Filters apply to the Summary, Anomaly review and Rules vs model tabs.")

# The accounts that match the filters, and the subsets the report talks about.
in_view = accounts[accounts["institution_code"].isin(chosen_banks)
                   & accounts["Client Type"].isin(chosen_client_types)]
flagged = in_view[in_view["anomaly_flag"]]           # the model thinks these look unusual
model_only = flagged[~flagged["has_exception"]]      # unusual, yet they pass every rule


# =============================================================================
# 4. HEADLINE NUMBERS (used in the Summary tab and in the Excel download)
# =============================================================================

simple_model = comparison.iloc[0]       # first row  = raw amounts only
improved_model = comparison.iloc[-1]    # last row   = engineered features
logistic = metrics[metrics["Model"] == "Logistic regression"].iloc[0]
baseline = metrics[metrics["Model"].str.startswith("Baseline")].iloc[0]

share_aware = (survey["Heard of DPF"] == "Yes").mean()
share_ready = (capstone_depositors["payout_status"] == "Ready to pay").mean()
protected_bn = capstone_depositors["protected"].sum() / 1e9


def percent_of(part, whole):
    """part / whole, safe when whole is 0 (e.g. every filter cleared)."""
    return part / whole if whole else 0


# =============================================================================
# 5. ONE FUNCTION PER TAB
# =============================================================================

def summary_tab():
    """Headline KPIs and the key findings, written as sentences for a report."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accounts checked", f"{len(in_view):,}")
    c2.metric("Flagged as unusual", f"{len(flagged):,}",
              f"{percent_of(len(flagged), len(in_view)):.1%} of accounts", delta_color="off")
    c3.metric("Unusual, passing every rule", f"{len(model_only):,}", "investigate first", delta_color="off")
    c4.metric("Known problems found (recall)", f"{improved_model['Recall %']:.0f}%",
              f"{improved_model['Recall %'] - simple_model['Recall %']:+.0f} pts vs simple model")
    c5.metric("Awareness model ROC AUC", f"{logistic['ROC AUC']:.2f}", "0.5 = coin toss", delta_color="off")

    st.subheader("Key findings")

    # The most common reason for a flag. Reasons look like "unusual X; unusual Y", so split them first.
    reasons = flagged["anomaly_reason"].str.split("; ").explode().value_counts()

    findings = [
        f"**{len(flagged):,} of {len(in_view):,} accounts** ({percent_of(len(flagged), len(in_view)):.1%}) "
        "were flagged as unusual for human review.",
        f"**{len(model_only):,} flagged accounts pass every rule.** "
        "These are the model's added value and should be investigated first.",
        f"Using engineered features, the model finds **{improved_model['Recall %']:.0f}%** of the known "
        f"numeric problems, compared with **{simple_model['Recall %']:.0f}%** using raw amounts only.",
    ]
    if len(reasons):
        findings.append(f"The most common reason for a flag is **{reasons.index[0]}** ({reasons.iloc[0]} accounts).")
    findings += [
        f"**{share_aware:.0%}** of survey respondents have heard of DPF. The awareness model ranks respondents "
        f"with ROC AUC **{logistic['ROC AUC']:.2f}**. Its accuracy is {logistic['Accuracy %']:.1f}%, against "
        f"{baseline['Accuracy %']:.1f}% for always guessing the majority answer.",
        f"Capstone run: **{len(capstone_accounts):,} accounts, {len(capstone_depositors):,} depositors**, "
        f"protected exposure **UGX {protected_bn:,.2f} bn**, {share_ready:.1%} of depositors ready to pay.",
    ]
    st.markdown("\n".join(f"{number}. {text}" for number, text in enumerate(findings, start=1)))
    st.info("A flagged record is a question, not a verdict. "
            "ML prioritises where people look; it does not replace the rules.")


def anomaly_tab():
    """The review queue: which accounts look unusual, and why."""
    if in_view.empty:
        st.warning("No accounts match the filters.")
        return

    # --- Controls ---
    lowest, highest = float(in_view["anomaly_score"].min()), float(in_view["anomaly_score"].max())
    # Start the slider at the model's own cut-off (the lowest score that was flagged).
    start = float(flagged["anomaly_score"].min()) if len(flagged) else highest
    threshold = st.slider("Show accounts with an anomaly score of at least", lowest, highest, start)
    only_model_finds = st.checkbox("Only accounts that pass every rule (model-only finds)")

    queue = in_view[in_view["anomaly_score"] >= threshold]
    if only_model_finds:
        queue = queue[~queue["has_exception"]]

    left, right = st.columns(2)

    # --- Chart 1: how the scores are spread, with the threshold marked (notebook 5.2 histogram) ---
    labelled = in_view.assign(Flag=in_view["anomaly_flag"].map({True: "Flagged for review", False: "Normal"}))
    histogram = px.histogram(labelled, x="anomaly_score", color="Flag", nbins=60, title="Anomaly score distribution",
                             color_discrete_map={"Flagged for review": COLOURS["critical"], "Normal": COLOURS["neutral"]})
    histogram.add_vline(x=threshold, line_dash="dash")
    left.plotly_chart(histogram, **WIDE)

    # --- Chart 2: the reasons given for the accounts in the queue ---
    reasons = queue["anomaly_reason"].replace("", pd.NA).dropna().str.split("; ").explode().value_counts()
    right.plotly_chart(bar_chart(reasons.sort_values(), "Why accounts were flagged", horizontal=True), **WIDE)

    # --- Chart 3: the queue by institution ---
    by_bank = queue.groupby("institution_code").size()
    left.plotly_chart(bar_chart(by_bank, "Accounts in this view, by institution"), **WIDE)

    # --- Chart 4: map of the branches those accounts belong to ---
    has_coordinates = {"Latitude", "Longitude"}.issubset(queue.columns) and queue["Latitude"].notna().any()
    if has_coordinates:
        right.plotly_chart(branch_map(queue), **WIDE)

    # --- The table itself, plus a download ---
    st.subheader(f"Review queue: {len(queue):,} accounts")
    table = queue.sort_values("anomaly_score", ascending=False)[REVIEW_COLUMNS]
    st.dataframe(table, hide_index=True, **WIDE)
    st.download_button("Download this review queue (CSV)", to_csv(table), "day5_anomaly_review_queue.csv", "text/csv")


def branch_map(queue):
    """A map with one bubble per branch; bubble size = number of accounts in the queue."""
    branches = (queue.dropna(subset=["Latitude", "Longitude"])
                .groupby(["Domicile Branch Name", "Latitude", "Longitude"]).size()
                .rename("Accounts").reset_index())

    # Plotly 5.24+ renamed scatter_mapbox -> scatter_map (and mapbox_style -> map_style).
    # Use whichever this computer has.
    if hasattr(px, "scatter_map"):
        make_map, style = px.scatter_map, {"map_style": "open-street-map"}
    else:
        make_map, style = px.scatter_mapbox, {"mapbox_style": "open-street-map"}

    fig = make_map(branches, lat="Latitude", lon="Longitude", size="Accounts", hover_name="Domicile Branch Name",
                   zoom=5.5, height=380, title="Branches of the accounts in this view",
                   color_discrete_sequence=[COLOURS["critical"]])
    return fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), **style)


def rules_vs_model_tab():
    """How the anomaly model compares with the 20 rules (notebook 5.2)."""
    left, right = st.columns(2)

    # The 2 x 2 table: rule exception yes/no  x  model unusual yes/no (same as the notebook's crosstab)
    two_by_two = pd.crosstab(in_view["has_exception"].map({True: "Rule exception", False: "No rule exception"}),
                             in_view["anomaly_flag"].map({True: "Model: unusual", False: "Model: normal"}))
    left.subheader("Rules vs model (2 × 2)")
    left.dataframe(two_by_two, **WIDE)
    left.caption("Bottom-right cell = unusual but passing every rule: the model's added value.")

    right.subheader("Simple vs improved anomaly model")
    right.dataframe(comparison, hide_index=True, **WIDE)

    # Recall and precision side by side for the two approaches
    long_format = comparison.melt(id_vars="Approach", value_vars=["Recall %", "Precision %"],
                                  var_name="Measure", value_name="%")
    st.plotly_chart(px.bar(long_format, x="Measure", y="%", color="Approach", barmode="group",
                           title="Finding the known numeric problems (whole dataset)",
                           color_discrete_sequence=[COLOURS["neutral"], COLOURS["dark"]]), **WIDE)

    st.subheader(f"Model-only finds: {len(model_only):,} accounts")
    st.dataframe(model_only.sort_values("anomaly_score", ascending=False)[
        ["Record ID", "institution_code", "Client Type", "Account Title/Name",
         "Account Currency", "Current Balance", "anomaly_reason"]], hide_index=True, **WIDE)


def awareness_tab():
    """The supervised model: who is likely to have heard of DPF (notebook 5.3)."""
    st.subheader("Awareness model: test-set results")
    st.dataframe(metrics, hide_index=True, **WIDE)

    left, right = st.columns(2)

    # --- Odds ratios: above 1 = more likely aware than the reference group ---
    chart_data = odds_ratios.assign(
        Direction=odds_ratios["Odds ratio"].map(lambda v: "More likely aware" if v >= 1 else "Less likely aware"))
    odds_chart = px.bar(chart_data, x="Odds ratio", y="Feature", orientation="h", color="Direction",
                        log_x=True, height=520, title="What predicts awareness of DPF? (1 = same as reference group)",
                        color_discrete_map={"More likely aware": COLOURS["ok"], "Less likely aware": COLOURS["critical"]})
    odds_chart.add_vline(x=1, line_color="black")
    left.plotly_chart(odds_chart, **WIDE)

    # --- Awareness % by region, one bar per survey wave (notebook 4.4) ---
    aware_by_region = (survey.assign(aware=survey["Heard of DPF"].eq("Yes"),
                                     wave=survey["Survey Wave"].astype(str))
                       .groupby(["Region", "wave"])["aware"].mean().mul(100).round(1)
                       .rename("Aware %").reset_index())
    right.plotly_chart(px.bar(aware_by_region, x="Region", y="Aware %", color="wave", barmode="group",
                              title="Awareness of DPF by region and survey wave", height=520,
                              labels={"wave": "Survey wave"},
                              color_discrete_sequence=[COLOURS["neutral"], COLOURS["dark"]]), **WIDE)

    # --- Score a profile with the trained model (notebook 5.3, "USE the model") ---
    saved = load_awareness_model()
    if saved is None:
        return
    st.subheader("Score a profile (for campaign planning)")
    boxes = st.columns(4)
    profile = {}
    for i, feature in enumerate(saved["features"]):
        choices = sorted(survey[feature].astype(str).unique())
        profile[feature] = boxes[i % 4].selectbox(feature, choices, key=feature)   # 4 dropdowns per row
    chance = saved["model"].predict_proba(pd.DataFrame([profile]))[0, 1]          # column 1 = "aware"
    st.metric("Predicted chance of being aware of DPF", f"{chance:.0%}")
    st.caption("Shows association, not cause. Use it to plan campaigns, "
               "never to treat individual depositors differently.")


def capstone_tab():
    """The full pipeline run on the capstone file (notebook section 5.5)."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{len(capstone_accounts):,}")
    c2.metric("Depositors", f"{len(capstone_depositors):,}")
    c3.metric("Accounts with an exception", f"{capstone_accounts['has_exception'].mean():.1%}")
    c4.metric("Protected exposure (UGX bn)", f"{protected_bn:,.2f}")

    left, right = st.columns(2)

    status = capstone_depositors.groupby(["institution_code", "payout_status"]).size().rename("Depositors").reset_index()
    left.plotly_chart(px.bar(status, x="institution_code", y="Depositors", color="payout_status",
                             title="Payout readiness by institution", color_discrete_map=STATUS_COLOURS), **WIDE)

    # Rule columns are named R01_..., R02_... ; each is True/False per account, so the sum is a count.
    rule_counts = capstone_accounts.filter(regex=r"^R\d\d_").sum().sort_values().tail(10)
    right.plotly_chart(bar_chart(rule_counts, "Top 10 failing rules", horizontal=True), **WIDE)

    protected_by_bank = (capstone_depositors.groupby("institution_code")["protected"].sum()
                         .div(1e9).round(3).rename("Protected (UGX bn)").reset_index())
    st.dataframe(protected_by_bank, hide_index=True, **WIDE)


def notebook_charts_tab():
    """Every chart the notebook drew (Days 1-5), shown under the notebook's own headings."""
    if not NOTEBOOK_FILE.exists():
        st.warning(f"Notebook not found: `{NOTEBOOK_FILE.name}`. Keep it in the same folder as this script.")
        return

    last_saved = NOTEBOOK_FILE.stat().st_mtime
    charts = load_notebook_charts(str(NOTEBOOK_FILE), last_saved)
    if not charts:
        st.warning("The saved notebook has no charts yet. Run it from the top, then save it (Ctrl + S).")
        return

    st.caption(f"{len(charts)} charts read from `{NOTEBOOK_FILE.name}`, "
               f"saved {datetime.fromtimestamp(last_saved):%Y-%m-%d %H:%M}. "
               "Interactive charts can be zoomed and hovered; the others are the notebook's own pictures.")

    # Warn if the notebook was saved BEFORE the latest results: its charts may be out of date.
    created = meta.get("created")
    if created and datetime.fromtimestamp(last_saved) < datetime.strptime(created, "%Y-%m-%d %H:%M"):
        st.warning("The notebook was saved before the latest results were created, so these charts may be "
                   "out of date. Save the notebook (Ctrl + S) and refresh this page.")

    # Choose a day; then optionally one section of that day.
    days = list(dict.fromkeys(chart["day"] for chart in charts))       # in notebook order, no repeats
    day = st.radio("Day", days, horizontal=True)
    of_day = [chart for chart in charts if chart["day"] == day]
    sections = list(dict.fromkeys(chart["section"] for chart in of_day))
    section = st.selectbox("Section", ["All sections"] + sections)
    if section != "All sections":
        of_day = [chart for chart in of_day if chart["section"] == section]

    # Draw them, with a heading whenever the section changes.
    current_section = None
    for number, chart in enumerate(of_day):
        if chart["section"] != current_section:
            current_section = chart["section"]
            st.subheader(current_section)
        if chart["kind"] == "plotly":
            # Rebuild the notebook's Plotly figure from its saved definition (skip settings this Plotly lacks).
            figure = pio.from_json(json.dumps(chart["chart"]), skip_invalid=True)
            st.plotly_chart(figure, key=f"nb_chart_{chart['cell']}_{number}", **WIDE)
        else:
            st.image(chart["chart"])
        st.caption(f"Notebook cell {chart['cell'] + 1}")


def downloads_tab():
    """The whole Day 5 report as one Excel workbook, and each table as CSV."""
    st.subheader("Download the Day 5 report")

    # The Excel report covers ALL accounts (it ignores the sidebar filters).
    all_flagged = accounts["anomaly_flag"]
    summary = pd.DataFrame({
        "Measure": ["Accounts checked", "Flagged as unusual", "Unusual but passing every rule",
                    "Anomaly recall (improved) %", "Awareness model ROC AUC", "Survey: aware of DPF %",
                    "Capstone accounts", "Capstone depositors", "Capstone protected (UGX bn)",
                    "Capstone ready to pay %"],
        "Value": [len(accounts), int(all_flagged.sum()), int((all_flagged & ~accounts["has_exception"]).sum()),
                  round(improved_model["Recall %"], 1), round(logistic["ROC AUC"], 3), round(share_aware * 100, 1),
                  len(capstone_accounts), len(capstone_depositors), round(protected_bn, 3),
                  round(share_ready * 100, 1)],
    })
    review_queue = (accounts[all_flagged].sort_values("anomaly_score", ascending=False)
                    .drop(columns=["Latitude", "Longitude"], errors="ignore"))

    sheets = {
        "Summary": summary,
        "Anomaly review queue": review_queue,
        "Rules vs model": results["rules_vs_model"],
        "Model comparison": comparison,
        "Awareness metrics": metrics,
        "Awareness odds ratios": odds_ratios,
        "Capstone depositors": capstone_depositors,
    }

    st.download_button("Download full report (Excel)", to_excel(sheets), "DPF_Day5_Report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("The Excel report ignores the sidebar filters and covers all accounts.")

    st.divider()
    for name, table in sheets.items():
        file_name = f"day5_{name.lower().replace(' ', '_')}.csv"
        st.download_button(f"{name} (CSV)", to_csv(table), file_name, "text/csv", key=f"csv_{name}")


# =============================================================================
# 6. THE PAGE
# =============================================================================

st.title("DPF Single Customer View - Day 5 report")
st.caption("Predictive analytics: anomaly detection, awareness model and capstone run. Dummy training data.")

tabs = st.tabs(["Summary", "Anomaly review", "Rules vs model", "Awareness", "Capstone run",
                "Notebook charts", "Downloads"])
tab_functions = [summary_tab, anomaly_tab, rules_vs_model_tab, awareness_tab, capstone_tab,
                 notebook_charts_tab, downloads_tab]

for tab, draw in zip(tabs, tab_functions):
    with tab:      # everything drawn inside this block appears on that tab
        draw()
