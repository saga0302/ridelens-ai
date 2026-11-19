"""Compare four detectors on faults the rule baseline does NOT check for.

Detectors (each fit only on the same-season training quarters, threshold set
at the 2nd percentile of training scores, FLAG_RATE = 0.02, as in train_season_matched.py):
  - rules          : plain checks (duration range, missing coords, speed > 60 km/h)
  - isolation_forest
  - lof            : Local Outlier Factor (novelty mode, 50k training sample)
  - hbos           : histogram-based outlier score, implemented here

Faults:
  covered by rules:   teleport_end_30km, impossible_speed, missing_end_coords,
                      duration_out_of_range, duration_zero
  NOT covered by rules: duration_x5_in_range, start_time_shift_6h, slow_far_trip

Caveat: we chose the faults, so this measures how well each detector catches
the faults we planted, not real-world anomalies.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_season_matched import discover, features, FEATURES  # noqa: E402
from evaluation.eval_injected import rule_flags, inject as inject_covered  # noqa: E402

TEST_QUARTERS = [(2026, 3), (2026, 2), (2026, 1), (2025, 4)]
N_CLEAN = 20_000
N_FAULT = 2_000
N_LOF_TRAIN = 50_000
N_HOLDOUT = 20_000
SEED = 42
FLAG_RATE = 0.02
HBOS_BINS = 50

COVERED = ["teleport_end_30km", "impossible_speed", "missing_end_coords",
           "duration_out_of_range", "duration_zero"]
UNCOVERED = ["duration_x5_in_range", "start_time_shift_6h", "slow_far_trip"]


def inject(df, fault):
    if fault in COVERED:
        return inject_covered(df, fault)
    out = df.copy()
    if fault == "duration_x5_in_range":
        out["duration"] = np.minimum(out["duration"] * 5, 1440)
    elif fault == "start_time_shift_6h":
        shifted = pd.to_datetime(out["start_time"], format="%m/%d/%Y %H:%M") - pd.Timedelta(hours=6)
        out["start_time"] = shifted.dt.strftime("%m/%d/%Y %H:%M")
    elif fault == "slow_far_trip":
        out["duration"] = 240.0
        out["end_lat"] = out["end_lat"] + 0.1
        out["end_lon"] = out["end_lon"] + 0.1
    else:
        raise ValueError(fault)
    return out


class HBOS:
    """Histogram-based outlier score: sum over features of -log(bin density)."""

    def fit(self, X):
        self.edges, self.logdens = [], []
        for j in range(X.shape[1]):
            counts, edges = np.histogram(X[:, j], bins=HBOS_BINS)
            dens = (counts + 1) / (counts.sum() + HBOS_BINS)  # Laplace smoothing
            self.edges.append(edges)
            self.logdens.append(np.log(dens / np.diff(edges)))
        return self

    def score(self, X):
        total = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            idx = np.clip(np.searchsorted(self.edges[j], X[:, j], side="right") - 1,
                          0, HBOS_BINS - 1)
            total += -self.logdens[j][idx]
        return total  # higher = more anomalous


def train_frame(data, ty, tq, train_keys):
    frames = []
    for k in train_keys:
        y, qq = int(k[:4]), int(k[-1])
        frames.append(data[(y, qq)])
    return pd.concat(frames, ignore_index=True)


def main():
    data = discover()
    rows = []
    for ty, tq in TEST_QUARTERS:
        bundle = joblib.load(f"models/season_q{tq}_for_{ty}Q{tq}.joblib")
        scaler, iforest, if_thr = bundle["scaler"], bundle["model"], bundle["threshold"]
        train_df = train_frame(data, ty, tq, bundle["train_keys"])
        Xtr = scaler.transform(features(train_df))
        rng = np.random.default_rng(SEED)

        # LOF: fit on a sample; threshold from a separate held-out training sample
        perm = rng.permutation(len(Xtr))
        lof_fit = Xtr[perm[:N_LOF_TRAIN]]
        hold = Xtr[perm[N_LOF_TRAIN:N_LOF_TRAIN + N_HOLDOUT]]
        lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(lof_fit)
        lof_thr = float(np.quantile(lof.score_samples(hold), FLAG_RATE))

        # HBOS: fit on all training rows; threshold from the same held-out sample
        hbos = HBOS().fit(Xtr)
        hbos_thr = float(np.quantile(hbos.score(hold), 1 - FLAG_RATE))

        def flags(df):
            X = scaler.transform(features(df))
            return {
                "rules": rule_flags(df),
                "isolation_forest": iforest.score_samples(X) < if_thr,
                "lof": lof.score_samples(X) < lof_thr,
                "hbos": hbos.score(X) > hbos_thr,
            }

        test = data[(ty, tq)]
        clean = test.sample(n=min(N_CLEAN, len(test)), random_state=SEED)
        row = {"test": f"{ty}-Q{tq}"}
        for name, f in flags(clean).items():
            row[f"fp__{name}"] = float(f.mean())

        for fault in COVERED + UNCOVERED:
            idx = rng.choice(len(test), size=min(N_FAULT, len(test)), replace=False)
            injected = inject(test.iloc[idx], fault)
            for name, f in flags(injected).items():
                row[f"recall__{name}__{fault}"] = float(f.mean())
        rows.append(row)

        print(f"\n{ty} Q{tq}  clean FP:  " + "  ".join(
            f"{n}={row[f'fp__{n}']:.1%}" for n in ["rules", "isolation_forest", "lof", "hbos"]))
        for fault in COVERED + UNCOVERED:
            tag = "covered" if fault in COVERED else "UNCOVERED"
            print(f"  {tag:<9} {fault:<22} " + "  ".join(
                f"{n}={row[f'recall__{n}__{fault}']:.1%}"
                for n in ["rules", "isolation_forest", "lof", "hbos"]))

    with open("results/evaluation/uncovered_eval_results.json", "w") as f:
        json.dump({"n_clean": N_CLEAN, "n_fault_per_type": N_FAULT, "seed": SEED,
                   "covered": COVERED, "uncovered": UNCOVERED, "results": rows},
                  f, indent=2)
    print("\nSaved results/evaluation/uncovered_eval_results.json")


if __name__ == "__main__":
    sys.exit(main())
