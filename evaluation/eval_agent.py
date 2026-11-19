"""Run the agent on the test cases and check each answer.

Usage (from the project root):
    python -m evaluation.eval_agent        # all cases
    python -m evaluation.eval_agent 3      # first 3 cases only, to control credit use
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

from agent.graph import build_graph  # noqa: E402
from evaluation.agent_cases import CASES  # noqa: E402


def score_answer(answer, grounded, case, unknown=()):
    """Return the list of failed checks. An empty list means the case passed."""
    failures = []
    if not grounded:
        failures.append(f"not grounded, numbers not in results: {list(unknown)}")
    for s in case.get("must", []):
        if s not in answer:
            failures.append(f"missing '{s}'")
    if case.get("must_any") and not any(s in answer for s in case["must_any"]):
        failures.append(f"none of {case['must_any']}")
    if case.get("cite") and not any(c in answer for c in case["cite"]):
        failures.append("no citation")
    for s in case.get("forbidden", []):
        if s.lower() in answer.lower():
            failures.append(f"forbidden '{s}'")
    for pattern in case.get("forbidden_regex", []):
        if re.search(pattern, answer):
            failures.append(f"forbidden pattern {pattern}")
    return failures


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASES)
    graph = build_graph()
    results = []
    for case in CASES[:limit]:
        out = graph.invoke({"question": case["question"]})
        failures = score_answer(out["answer"], out.get("grounded", False), case,
                                out.get("unknown_numbers", []))
        results.append({"question": case["question"], "passed": not failures,
                        "failures": failures, "answer": out["answer"]})
        print(f"{'PASS' if not failures else 'FAIL'}  {case['question']}")
        for f in failures:
            print(f"      - {f}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{passed} of {len(results)} passed")

    out_dir = os.path.join(ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "agent_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'agent_eval.json')}")


if __name__ == "__main__":
    main()