"""Save one summary file per quarter, so the app can run without the raw trip files.

For each quarter in data/raw it saves:
  data/summaries/YYYY-QN.json            same-quarter numbers, checksum and row count
  data/summaries/profiles/YYYY-QN.json   typical values used for trip reasons,
                                         from earlier same-season quarters

Usage (from the project root):  python -m pipeline.build_summaries
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison.reasons import profile  # noqa: E402
from comparison.same_quarter import FIRST_YEAR, summarise  # noqa: E402
from pipeline.normalize import load_quarter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "summaries")
PROFILES = os.path.join(OUT, "profiles")
MIN_PAST = 3


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    os.makedirs(PROFILES, exist_ok=True)
    frames = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*-Q*"))):
        m = re.search(r"(\d{4})-Q(\d)$", d)
        if not m:
            continue
        y, q = int(m.group(1)), int(m.group(2))
        csv = glob.glob(os.path.join(d, "*.csv"))[0]
        df, _ = load_quarter(csv)
        frames[(y, q)] = df
        record = {"year": y, "quarter": q, "rows": int(len(df)),
                  "sha256": _sha256(csv), "summary": summarise(df)}
        with open(os.path.join(OUT, f"{y}-Q{q}.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"saved summary {y}-Q{q}")

    for (y, q) in sorted(set(frames) | {(2026, 4)}):
        past = {k: v for k, v in frames.items() if k[1] == q and FIRST_YEAR <= k[0] < y}
        if len(past) < MIN_PAST:
            continue
        med, iqr, shares = profile(past)
        record = {"median": med.to_dict(), "iqr": iqr.to_dict(),
                  "pass_shares": shares.to_dict(),
                  "past_quarters": [f"{k[0]}-Q{k[1]}" for k in sorted(past)]}
        with open(os.path.join(PROFILES, f"{y}-Q{q}.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"saved profile {y}-Q{q}")


if __name__ == "__main__":
    main()