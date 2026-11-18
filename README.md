# RideLens AI: LA Metro bike-share quarterly change tracker

When a new quarterly trip file is published, this project checks it, compares it
with the same quarter in earlier years, and flags changes that look real.

Status: **work in progress.** Data checks and models are built and tested. The
LangGraph agent, RAG, MCP tools, Streamlit dashboard and Airflow schedule are not
started.

## Layout

```
data/          raw CSVs (not in git) and manifest.json (gate results per quarter)
pipeline/      clean and check the data: normalize.py, quality_gate.py, run_gate_all.py
training/      season-matched Isolation Forest: train_season_matched.py
evaluation/    fault-injection tests: eval_injected.py, eval_uncovered.py, quarter_hour_check.py
models/        trained model files (.joblib), one per test quarter
results/       every evaluation result as .json
legacy/        first model (mixed seasons), superseded, kept for history
tests/         gate tests (11)
```

## Key results so far

- **Season-matched model:** trained on 2019 onward. Flag rates on the four test
  quarters are 5.0–5.9%, against a 5% target.
- **Injected faults the rules catch:** rules catch 81–100%; Isolation Forest catches 4–100%.
- **Injected faults the rules miss:** Isolation Forest catches a slow, long trip
  (~98%) but only 9–17% of a 6-hour time shift or a 5× duration inflation. LOF and
  HBOS do no better on those.
- **Quarter-level hour check:** catches a 6-hour time shift once it touches about
  20% of trips; misses shifts of 5% or less.

These faults are chosen by us, so the numbers show how each method handles the
faults we planted, not how often real anomalies are missed. There are no labelled
real anomalies yet.

## Data

Raw trip files are **not in this repo** (about 450 MB). Put each quarter's trip CSV in
`data/raw/<YYYY>-Q<q>/`, for example `data/raw/2026-Q3/metro-trips-2026-q3.csv`.
The 39 quarters from 2017 Q1 to 2026 Q3 come from the public LA Metro bike-share
archive. Only quarters from 2019 onward are used for training.

## Setup

Use Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run everything from the project root.

```bash
python -m pytest -q tests                        # gate tests
python pipeline/run_gate_all.py                  # gate every quarter, writes data/manifest.json
python training/train_season_matched.py          # season-matched models (a few minutes)
python evaluation/eval_injected.py               # rules vs Isolation Forest, planted faults
python evaluation/eval_uncovered.py              # adds LOF and HBOS, faults rules miss
python evaluation/quarter_hour_check.py          # quarter-level hour-of-day check
```

Results are written to `results/`.

## Design

The full plan is in the project doc `claude/Quarterly_Tracker_Design.md`.
