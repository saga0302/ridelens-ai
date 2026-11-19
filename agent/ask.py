"""Ask the RideLens agent a question.

Usage (from the project root):
    python -m agent.ask "How many trips were flagged in 2026 Q3?"

The API key is read from a .env file in the project root (git-ignored).
"""
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

from agent.graph import build_graph  # noqa: E402

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project root.")
    graph = build_graph()
    out = graph.invoke({"question": " ".join(sys.argv[1:])})
    print(out["answer"])
    print(f"\n(grounded: {out.get('grounded')})")