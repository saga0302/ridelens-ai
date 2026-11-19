"""Print the same-quarter comparison for one quarter.

Usage (from the project root):  python -m comparison.run_same_quarter 2026 3
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.normalize import load_quarter  # noqa: E402
from comparison.same_quarter import FIRST_YEAR, compare_quarter  # noqa: E402


def load(year, q):
    csvs = glob.glob(os.path.join(ROOT, "data", "raw", f"{year}-Q{q}", "*.csv"))
    df, _ = load_quarter(csvs[0])
    return df


if __name__ == "__main__":
    year, q = int(sys.argv[1]), int(sys.argv[2])
    frames = {}
    for y in range(FIRST_YEAR, year + 1):
        if glob.glob(os.path.join(ROOT, "data", "raw", f"{y}-Q{q}", "*.csv")):
            frames[(y, q)] = load(y, q)
    report = compare_quarter(frames, year, q)
    print(report.round(4).to_string(index=False))