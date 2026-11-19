# RideLens AI: LA Metro bike-share quarterly change tracker

When a new quarterly trip file is published, this project checks it, compares it
with the same quarter in earlier years, and explains what changed. It's built for
people who use the data in their own projects: they get a dashboard, and a chatbot
explains what it shows.

Status: **work in progress.** Data checks, the season-matched model, the fault
comparison and the same-quarter report are built and tested. Still to build: the
quarter-level alert rule, station-level views, the dashboard, the LangGraph agent,
RAG, MCP tools and the Airflow schedule.

## Layout

```
data/          raw CSVs (not in git) and manifest.json (gate results per quarter)
pipeline/      clean and check the data: normalize.py, quality_gate.py, run_gate_all.py
training/      season-matched Isolation Forest: train_season_matched.py
evaluation/    fault comparison and quarter-level check
comparison/    same-quarter comparison and the per-quarter report
models/        trained model files (.joblib), one per test quarter
results/       training/ (season-matched model), evaluation/ (fault comparison,
               quarter-level check), reports/ (one report per quarter)
tests/         19 tests: gate (11), same-quarter comparison (5), report (3)
```

## How the cutoff is set

A trip is flagged when its Isolation Forest score falls below a cutoff. The cutoff
comes from past data, not from a value chosen up front:

- For each past quarter, train on the other same-season quarters from 2019 onward,
  then score the held-out quarter. This gives out-of-sample scores for every quarter.
- The cutoff is the 2nd percentile (2%) of those pooled out-of-sample scores.
- A z-score cutoff (for example, −3 standard deviations, or the robust version using
  the median and MAD) was tested first. On these skewed scores it flagged almost no
  trips, so it was dropped.
- Check on the four test quarters: flag rates were 2.1–2.6% against a 2% target.
  An earlier 5% cutoff gave 5.0–5.9%.

## Same-quarter comparison

Each quarter's summary numbers are compared with the same quarter in every past year
from 2019 on: median trip length and distance, mean start hour, trip count, weekend
and round-trip shares, and pass-type shares.

- Each metric has a range from the past years, and a cutoff from leave-one-year-out
  gaps. Both are shown as information.
- These numbers do not raise alerts. They describe where the quarter sits against
  past years.

## Alerts (decided so far)

- **Data alerts:** a quarter that fails the data gate is not analysed. The report
  shows which checks failed.
- **Pattern information:** same-quarter numbers are shown with the past range.
  No alert is attached.
- **Trip examples:** flagged trips are listed, the most unusual first. A flagged trip
  is not an alarm on its own, since many will turn out to be data errors.
- **Quarter-level rule:** not decided yet. Options are a past-range rule tested on
  past quarters first, or report only until real uploads are checked.

## Who it's for and what it covers

- Built for people using the data in their own projects: a dashboard, with a chatbot
  that explains it.
- Station data is in scope. Weather is out of scope for now.
- The app describes patterns and possible causes. It does not label an area as
  unsafe, since trip data cannot show that.

## Key results so far

- **Season-matched model:** trained on 2019 onward, with the 2% held-out cutoff.
  Flag rates on the four test quarters are 2.1–2.6%.
- **Clean data:** at the 2% cutoff, Isolation Forest flags 2.1–2.5% of clean trips.
- **Faults the rules catch:** the rules catch 81–100% of the faults they cover.
  Isolation Forest catches 64–86% of the out-of-range durations but only 1–30% of the
  other covered faults.
- **Faults the rules miss:** Isolation Forest catches the slow, far trip (95–97%). No
  method catches more than 8.5% of the 5× duration inflation or more than 6.4% of the
  6-hour time shift, except LOF on the 5× inflation (up to 24%).
- **Quarter-level hour check:** catches a 6-hour time shift once it touches about 20%
  of trips, and misses shifts of 5% or less. Clean files stay under the threshold in all
  four test quarters.
- **2026 Q3 report:** 2.23% of trips flagged against the 2% expectation. No same-quarter
  metric is outside its cutoff.
- **Open issue:** the trip-level rules flag 3.0–4.8% of clean trips. Most of this comes
  from missing end coordinates, which the gate allows up to 5%. Decide whether missing
  coordinates should be a trip flag or a data-quality warning.

These faults are chosen by us, so the numbers show how each method handles the faults
we planted, not how often real anomalies are missed. There are no labelled real
anomalies yet.

## Data

Raw trip files are **not in this repo** (about 430 MB for 2019 onward). Put each
quarter's trip CSV in `data/raw/<YYYY>-Q<q>/`, for example
`data/raw/2026-Q3/metro-trips-2026-q3.csv`.

Training uses 2019 Q1 to 2026 Q3, which is 31 quarters. Earlier quarters were removed
because the 2017–2018 data is from an older system and was not used.

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
python -m pytest -q tests                        # all tests (19)
python pipeline/run_gate_all.py                  # gate every quarter, writes data/manifest.json
python training/train_season_matched.py          # season-matched models
python evaluation/eval_uncovered.py              # fault comparison: rules, IF, LOF, HBOS
python evaluation/quarter_hour_check.py          # quarter-level hour-of-day check
python evaluation/threshold_check.py             # cutoff study: z-score vs percentile
python -m comparison.run_same_quarter 2026 3     # same-quarter table for one quarter
python -m comparison.run_report 2026 3           # full report for one quarter, saved to results/reports/
```

## Design

