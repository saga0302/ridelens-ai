"""Build and save one quarter's report.

Usage (from the project root):  python -m comparison.run_report 2026 3
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from comparison.report import build_report  # noqa: E402
from comparison.run_same_quarter import load  # noqa: E402
from comparison.same_quarter import FIRST_YEAR  # noqa: E402

if __name__ == "__main__":
    year, q = int(sys.argv[1]), int(sys.argv[2])
    frames = {}
    for y in range(FIRST_YEAR, year + 1):
        if glob.glob(os.path.join(ROOT, "data", "raw", f"{y}-Q{q}", "*.csv")):
            frames[(y, q)] = load(y, q)

    report = build_report(frames, year, q)

    out = os.path.join(ROOT, "results", "reports", f"report_{year}Q{q}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    t = report["trip_level"]
    print(f"{report['quarter']}  gate passed: {report['data_quality']['gate_passed']}")
    print(f"trips {t['trips']:,}  flagged {t['flagged']:,}  "
          f"share {t['flagged_share']:.2%}  expected {t['expected_share']:.0%}")
    print(f"saved {out}")