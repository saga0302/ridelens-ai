"""Plain-language reasons for flagged trips.

For each flagged trip, names the features that are most unusual compared with
past trips in the same season. These describe the trip; they are not causes.
"""
import numpy as np
import pandas as pd

from training.train_season_matched import features

NUMERIC = {"duration_min": ("trip length", "min", 0),
           "distance_km": ("distance", "km", 1),
           "start_hour": ("start hour", "h", 0)}
Z_MIN = 3
RARE_SHARE = 0.05


def profile(past_frames):
    """Typical value, spread and pass-type shares over past same-season trips."""
    X = pd.concat([features(df) for df in past_frames.values()], ignore_index=True)
    num = X[list(NUMERIC)]
    med = num.median()
    iqr = (num.quantile(0.75) - num.quantile(0.25)).replace(0, np.nan)
    shares = X[[c for c in X.columns if c.startswith("pass_")]].mean()
    return med, iqr, shares


def reasons(df, idx, prof):
    med, iqr, shares = prof
    row = features(df.loc[[idx]]).iloc[0]
    found = []
    for name, (label, unit, dp) in NUMERIC.items():
        if pd.isna(iqr[name]):
            continue
        z = (row[name] - med[name]) / iqr[name]
        if abs(z) >= Z_MIN:
            found.append((abs(z), f"{label} {row[name]:.{dp}f} {unit} (typical {med[name]:.{dp}f})"))
    for col in shares.index:
        if row[col] == 1 and shares[col] < RARE_SHARE:
            found.append((10.0, f"uncommon passholder type: {col[5:]} ({shares[col]:.1%} of past trips)"))
    if pd.isna(df.loc[idx, "end_lat"]) or pd.isna(df.loc[idx, "end_lon"]):
        found.append((20.0, "end location missing (possible data issue)"))
    found.sort(key=lambda t: -t[0])
    return [text for _, text in found[:2]] or ["unusual combination of values"]