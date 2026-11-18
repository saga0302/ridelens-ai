# LA Metro Bike Share: quarterly change tracker

When a new quarterly trip file is published, this project checks it, compares it
with the same quarter in earlier years, and flags changes that look real.

Status: **work in progress.** Data checks and models are built and tested. The
LangGraph agent, RAG, MCP tools, Streamlit dashboard and Airflow schedule are not
started.

## What is here

| Piece | File | Status |
|---|---|---|
| Normalise every era of trip files to one schema | `normalize.py` | done, used by all steps below |
| Quarterly quality gate (plain code, hard checks) | `quality_gate.py`, `run_gate_all.py` | done, 37 of 39 quarters pass; 2020 Q3 and 2020 Q4 fail on missing values (flagged, kept) |
| Gate tests (fault injection) | `tests/test_quality_gate.py` | 11 passing |
| Season-matched Isolation Forest | `models/train_season_matched.py` | done; trains on 2019 onward, same quarter-of-year only |
| Injected-fault evaluation: rules vs Isolation Forest | `models/eval_injected.py` | done |
| Uncovered-fault evaluation: rules vs Isolation Forest vs LOF vs HBOS | `models/eval_uncovered.py` | done |
| Quarter-level hour-of-day check | `models/quarter_hour_check.py` | done |
| Superseded first model (mixed seasons, kept for history) | `models/train_trip_model.py`, `models/trip_iforest_v1.*` | superseded; the script's paths no longer match the data layout |

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
`data/raw/<YYYY>-Q<q>/` (for example `data/raw/2026-Q3/metro-trips-2026-q3.csv`).
The 39 quarters from 2017 Q1 to 2026 Q3 come from the public LA Metro bike-share
archive. Only quarters from 2019 onward are used for training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m pytest -q tests                  # gate tests
python run_gate_all.py                     # gate every quarter, writes data/manifest.json
python models/train_season_matched.py      # season-matched models (a few minutes)
python models/eval_injected.py             # rules vs Isolation Forest, planted faults
python models/eval_uncovered.py            # adds LOF and HBOS, faults rules miss
python models/quarter_hour_check.py        # quarter-level hour-of-day check
```

Results are written to `models/*.json`.

## Design

The full plan is in the project doc `claude/Quarterly_Tracker_Design.md`.
