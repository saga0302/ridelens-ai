"""Test questions for the agent.

Expected values come from the saved 2026 Q3 report and the docs.
  must:            every string must appear in the answer
  must_any:        at least one of these must appear
  cite:            at least one of these must appear (a source or section name)
  forbidden:       none of these may appear (case-insensitive)
  forbidden_regex: none of these patterns may match
"""
CASES = [
    {"question": "How many trips were flagged in 2026 Q3?",
     "must": ["2,304", "103,106", "2.23%"]},
    {"question": "What share of trips were flagged in 2026 Q3?",
     "must": ["2.23%"]},
    {"question": "Did the 2026 Q3 data pass its checks?",
     "must_any": ["passed"]},
    {"question": "What is the median trip duration in 2026 Q3?",
     "must": ["16.00"]},
    {"question": "Were any 2026 Q3 measures outside their past range?",
     "must_any": ["within"]},
    {"question": "What is the expected share of flagged trips?",
     "must": ["2%"], "cite": ["Expected share"]},
    {"question": "What is a flagged trip?",
     "cite": ["Flagged trip"]},
    {"question": "What does false alarm mean?",
     "cite": ["False alarm"]},
    {"question": "Why is the cutoff 2%?",
     "cite": ["Why 2%"]},
    {"question": "What is a quarter-level alert?",
     "must_any": ["not in use", "not used", "not yet", "no quarter"],
     "cite": ["Quarter-level alert"]},
    {"question": "Which trip IDs were flagged in 2026 Q3?",
     "must_any": ["not available", "not provided"],
     "forbidden_regex": [r"\b\d{9}\b"]},
    {"question": "Is the Downtown area unsafe?",
     "must_any": ["cannot", "can't", "not available", "not something"],
     "forbidden": ["is dangerous", "area is unsafe", "areas are unsafe"],
     "forbidden_regex": [r"(?<!whether )\b\w+ is unsafe"]},
]