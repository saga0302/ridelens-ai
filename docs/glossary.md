# Glossary

## Flagged trip
A trip the model scores as unusual: its score falls below the cutoff. A flag is a
reason to look, not proof of a problem. Many flagged trips turn out to be data errors.

## Cutoff
The score below which a trip is flagged. It is the 2nd percentile (2%) of
out-of-sample scores from past quarters, set before the new quarter is seen.

## Out-of-sample score
A score from a model that never saw the quarter being scored. This keeps the
cutoff honest, because the quarter is never judged by a model trained on it.

## Leave-one-quarter-out
Each past training quarter is scored by a model trained on the other same-season
quarters. The scores from these runs give the cutoff.

## Season-matched
Each quarter is compared only with the same quarter of the year, such as Q3 with
earlier Q3s, because trip patterns change with the season.

## Flagged share
Flagged trips divided by all trips in the quarter.

## Expected share
The share of trips the model flags in normal conditions, by design: 2%.

## Within past range, outside past range
Whether a summary number for this quarter lies between the lowest and highest
value seen in past same-quarter years. Being inside the range means the quarter
looks like past quarters on that measure.

## Same-quarter comparison
Each summary number for a quarter (median trip length, pass mix, weekend share and
so on) set against the same quarter in past years. It is shown as information and
does not raise an alert.

## Data-quality gate
Eleven checks run before a quarter is analysed: schema, timestamps, coverage,
duplicate trip IDs, duration range, pass types and missing coordinates (at most 5%).
A quarter that fails is not analysed, and the report says which checks failed.

## Planted fault
A deliberate error inserted into clean data to test whether a method catches it.
Planted faults are used for evaluation only, never for training.

## Detectors
The methods compared on planted faults: simple rules, Isolation Forest, Local
Outlier Factor (LOF) and histogram-based outlier scores (HBOS). The model that
produces flags is Isolation Forest.

## Backtest
Replaying the quarter-level rule on past quarters to see how often it would have
fired.

## False alarm
An alert on a quarter that is in fact normal. There are no labelled anomalies, so
fired alerts on past quarters are counted as possible false alarms.

## Quarter-level alert
Not in use. Quarters are reported, not alerted, until real uploads show what
normal looks like.