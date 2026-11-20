# RideLens AI: LA Metro bike-share quarterly tracker

![CI](https://github.com/saga0302/ridelens-ai/actions/workflows/ci.yml/badge.svg)

**Live app:** https://ridelens-ai-5obhdki3aukh3koxypmjdy.streamlit.app

When LA Metro publishes a new quarter of bike-share trips, RideLens checks the data, compares the quarter with the same quarter in earlier years, flags unusual trips, and explains the results in a dashboard with a chat assistant.

## Screenshots

![Overview](docs/screenshots/Overview.png)
![Compare](docs/screenshots/Compare.png)
![Ask](docs/screenshots/Ask.png)
![Upload](docs/screenshots/Upload.png)

## How it works

```mermaid
flowchart LR
    classDef src fill:#070712,stroke:#00f0ff,stroke-width:2px,color:#e8e8ff
    classDef proc fill:#070712,stroke:#ff2bd6,stroke-width:2px,color:#e8e8ff
    classDef store fill:#070712,stroke:#b6ff3b,stroke-width:2px,color:#e8e8ff
    classDef ai fill:#070712,stroke:#ff2bd6,stroke-width:3px,color:#b6ff3b
    classDef ui fill:#070712,stroke:#00f0ff,stroke-width:3px,color:#00f0ff

    M["LA Metro quarterly trip CSVs<br/>2019 Q1 to 2026 Q3"]:::src
    G["Quality gate<br/>11 checks"]:::proc
    S[("Per-quarter summaries<br/>and checksums")]:::store
    T["Season-matched Isolation Forest<br/>2% cutoff from past data"]:::proc
    MD[("Saved models<br/>one per quarter")]:::store
    C["Same-quarter comparison<br/>information only"]:::proc
    R[("Saved reports<br/>JSON per quarter")]:::store
    E["Evaluation<br/>planted faults and backtest"]:::proc
    TOOLS["Read-only tools<br/>and MCP server"]:::proc
    RAG[("RAG index<br/>ChromaDB")]:::store
    AG["LangGraph agent<br/>keyword routing"]:::ai
    CL["Claude<br/>writes the answer"]:::ai
    GC["Grounding check<br/>numbers vs results"]:::ai
    APP["Streamlit dashboard<br/>7 tabs"]:::ui
    UP["Upload<br/>2026 Q3 onward"]:::ui

    M --> G --> S
    S --> T --> MD
    S --> C --> R
    M -.-> E
    R --> TOOLS --> AG
    RAG --> AG --> CL --> GC --> APP
    R --> APP
    MD --> APP
    UP --> G
    UP --> MD
    APP --> AG
```

## What it does

- Checks each quarterly file against 11 data-quality rules before anything else runs.
- Compares each quarter only with the same season in earlier years, from 2019 onward, because trip patterns change with the season.
- Flags about 2% of trips as unusual. The cutoff comes from past data, not from a fixed rule of thumb.
- Shows same-quarter differences as information only. Trip-level flags are the only alerts.
- Answers questions in plain English. Answers come from saved results, and every number is checked against them.

## Results so far

| Quarter | Trips | Flagged | Share | Data checks |
|---|---|---|---|---|
| 2026-Q1 | 88,932 | 2,292 | 2.58% | passed |
| 2026-Q2 | 104,654 | 2,228 | 2.13% | passed |
| 2026-Q3 | 103,106 | 2,304 | 2.23% | passed |

The expected share is 2% by design. A quarter above 2% has more unusual trips than past quarters. That's a reason to look, not a finding.

## How the cutoff is set

For each past quarter, a model is trained on the other quarters from the same season and scores the quarter it didn't see. Pooling those scores gives a cutoff at the 2nd percentile, so about 2% of past trips fall below it. A z-score cutoff was tested and rejected, because on these scores it flagged almost nothing.

## Testing the method

- **Planted faults.** We added known problems to test data (slow trips, time shifts, inflated durations) and checked which methods catch them. The rules catch most of the faults they were written for. Isolation Forest catches the slow-trip fault (about 98%), but only 9 to 17% of the time shift and duration inflation. No per-trip method catches a whole-quarter time shift. Planted faults are used for testing only, never for training.
- **Quarter-level alert.** A backtest of the rule "alert when a quarter is above every earlier quarter" fired in 2 of 15 quarters, where chance alone predicts 3.6. That's why quarter-level alerts are report-only.

## Dashboard

Overview, Compare, Trips, Trend, Methods, Ask, About and Upload.

## The chat

The Ask tab is protected by a passcode and allows 10 questions per session. The assistant answers from the saved results and the project documentation. It won't say whether an area is safe, and it won't say why a particular trip happened.

## Limits

- Not affiliated with LA Metro. It uses LA Metro's public trip files.
- A flagged trip is unusual, not proven wrong. Many are likely data errors or rare trips.
- The planted faults are our own design. There are no labelled real anomalies to test against yet.
- Trip-level data only. No rider information, and no judgement of any area's safety.
- Uploads are accepted for 2026 Q3 onward, and only for complete quarters.

## Layout

- `data/`: trip files (not in git, except small summaries) and `manifest.json` with the quality results per quarter
- `pipeline/`: loading, normalising, checking and summarising the data
- `training/`: the season-matched Isolation Forest
- `evaluation/`: planted-fault comparison, quarter-level check and backtest, agent tests
- `comparison/`: same-quarter comparison and the per-quarter report
- `models/`: saved models, one per quarter
- `results/`: saved training, evaluation and report outputs
- `rag/`: the document index for the assistant
- `agent/`: the question-answering agent
- `tools/`: read-only functions over the saved results
- `mcp_server/`: the same tools exposed over MCP
- `dashboard/`: charts and styling for the Streamlit app
- `docs/`: the glossary and dashboard screenshots
- `tests/`: the test suite
- `app.py`: the dashboard

## Setup

Use Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The live app doesn't need the raw trip files. To rebuild everything from scratch, download the LA Metro trip files and place each in `data/raw/YYYY-QN/`, for example `data/raw/2026-Q3/`.

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key
APP_PASSCODE=your-passcode
CONTACT_EMAIL=your-address
```

## Run

```
python pipeline/run_gate_all.py           # quality checks for all quarters
python training/train_season_matched.py   # train the models
python -m pipeline.build_summaries        # save per-quarter summaries
python -m comparison.run_report 2026 3    # build one quarter's report
streamlit run app.py                      # open the dashboard
```

## Tests

```
python -m pytest -q                       # 47 tests
python -m evaluation.eval_agent           # 13 assistant cases (uses the API)
```

## Next

- Feedback on flagged trips, stored outside the app
- Station-level views
- Automatic quarterly updates
