"""Compare a new quarter with the same quarter in past years.

For each metric, the cutoff is the largest leave-one-year-out deviation among the
past same-quarter years. A year is compared with the median of the other years, so
the cutoff comes from past data, not from a value chosen up front.
"""
import pandas as pd

from training.train_season_matched import FEATURES, features

FIRST_YEAR = 2019
MIN_PAST_YEARS = 3


def summarise(df):
    """One row of summary numbers for a normalised quarter."""
    X = features(df)
    out = {
        "median_duration_min": float(X["duration_min"].median()),
        "median_distance_km": float(X["distance_km"].median()),
        "mean_start_hour": float(X["start_hour"].mean()),
        "trips": float(len(X)),
    }
    for col in ["is_weekend", "is_round_trip"] + [c for c in FEATURES if c.startswith("pass_")]:
        out[f"share_{col}"] = float(X[col].astype(float).mean())
    return out


GAP_FLOOR = 0.01

def _relative_gap(value, base):
    return abs(value - base) / max(abs(base), GAP_FLOOR)

def compare_summaries(new, past):
    """new: dict of metric -> value for the new quarter.
    past: DataFrame, one row per past year, one column per metric."""
    if len(past) < MIN_PAST_YEARS:
        raise ValueError(f"need at least {MIN_PAST_YEARS} past years, got {len(past)}")

    rows = []
    for metric in past.columns:
        vals = past[metric]
        # Cutoff: largest gap between one year and the median of the other years.
        loo_gaps = []
        for year in vals.index:
            others = vals.drop(year)
            loo_gaps.append(_relative_gap(vals[year], others.median()))
        cutoff = max(loo_gaps)

        base = vals.median()
        gap = _relative_gap(new[metric], base)
        rows.append({
            "metric": metric,
            "new": new[metric],
            "past_median": base,
            "past_min": vals.min(),
            "past_max": vals.max(),
            "gap": gap,
            "cutoff": cutoff,
            "flagged": bool(gap > cutoff),
        })
    return pd.DataFrame(rows)


def compare_quarter(frames, year, q):
    """frames: dict {(year, quarter): normalised DataFrame}.
    Compares (year, q) with the same quarter in every past year from FIRST_YEAR."""
    new = summarise(frames[(year, q)])
    past_years = [y for y in range(FIRST_YEAR, year) if (y, q) in frames]
    past = pd.DataFrame({y: summarise(frames[(y, q)]) for y in past_years}).T
    return compare_summaries(new, past)