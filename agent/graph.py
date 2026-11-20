"""RideLens question-answering agent (LangGraph).

Routes a question to the saved numbers (tools), the written explanations (RAG),
or both. Claude writes the answer from those results only. A grounding check then
confirms every number in the answer appears in the results; if one does not, the
answer is replaced with the raw results.
"""
import json
import os
import re
import sys
from typing import TypedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langgraph.graph import END, StateGraph  # noqa: E402
from rag.retrieve import search  # noqa: E402
from tools import quarter_tools as qt  # noqa: E402

MODEL = "claude-haiku-5-5"
QUARTER_RE = re.compile(r"(20\d\d)\s*-?\s*Q([1-4])", re.IGNORECASE)
NUMBER_WORDS = ["how many", "share", "percent", "trip", "median", "flagged", "pass",
                "passed", "failed", "gate", "check", "compare", "range", "quarter", "report"]
METHOD_WORDS = ["why", "what is", "what does", "how does", "how is", "explain", "mean",
                "cutoff", "percentile", "method", "alert", "false alarm", "backtest", "glossary"]
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
SYSTEM_PROMPT = """You explain RideLens results to people using LA Metro bike-share data.
Rules:
- Use only the FACTS and PASSAGES provided. If something is not there, say "not available".
- Use only numbers that appear in FACTS or PASSAGES. Write counts in words, e.g. "seven past years".
- Quote numbers exactly as written in FACTS. Do not round or calculate new numbers.
- Say "unusual compared with past years", not "wrong", unless data_checks says failed.
- When asked what could be behind something, list possible causes as guesses labelled "possible, not confirmed", and name the check that would confirm or rule out each one.
- Never state a cause as fact. Never label an area as unsafe.
- Do not mention trip IDs; they are not provided.
- Cite numbers to that quarter's report, e.g. [2026-Q2 report]. Cite definitions to the glossary or README, e.g. [glossary: Cutoff]. Never cite a report for a definition, or the glossary for a number.
- Keep the answer short and plain."""


class AgentState(TypedDict, total=False):
    question: str
    need_numbers: bool
    need_docs: bool
    facts: dict
    passages: list
    answer: str
    grounded: bool
    unknown_numbers: list


def _fmt_metric(name, value):
    if name.startswith("share_"):
        return f"{value:.2%}"
    if name == "trips":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _quarter_from(question):
    m = QUARTER_RE.search(question)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    saved = [r["quarter"] for r in qt.list_quarters() if r["report_saved"]]
    return saved[-1] if saved else None


def route(state):
    q = state["question"].lower()
    quarter_mentioned = bool(QUARTER_RE.search(state["question"]))
    need_numbers = quarter_mentioned or any(w in q for w in NUMBER_WORDS)
    need_docs = any(w in q for w in METHOD_WORDS) or not need_numbers
    return {"need_numbers": need_numbers, "need_docs": need_docs}


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve"]


def get_numbers(state):
    quarter = _quarter_from(state["question"])
    if quarter is None:
        return {"facts": {"error": "no saved report is available"}}
    summary = qt.get_trip_summary(quarter)
    if "error" in summary:
        return {"facts": {"quarter": quarter, "error": summary["error"]}}
    gate = qt.get_gate_result(quarter)
    same = qt.get_same_quarter(quarter)

    if gate.get("gate_passed"):
        checks = "passed all data checks"
    else:
        failed = [f["check"] for f in gate.get("gate_failures", [])]
        checks = "failed: " + ", ".join(failed)

    rows = []
    for m in same["metrics"]:
        where = "outside" if m["outside_past_range"] else "within"
        rows.append(
            f"{m['metric']}: this quarter {_fmt_metric(m['metric'], m['new'])}, "
            f"past median {_fmt_metric(m['metric'], m['past_median'])}, "
            f"past range {_fmt_metric(m['metric'], m['past_min'])} to "
            f"{_fmt_metric(m['metric'], m['past_max'])}, {where} past range"
        )

    outside = [m["metric"] for m in same["metrics"] if m["outside_past_range"]]
    n = len(same["metrics"])
    facts = {
        "quarter": quarter,
        "trips": f"{summary['trips']:,}",
        "flagged": f"{summary['flagged']:,}",
        "flagged_share": f"{summary['flagged_share']:.2%}",
        "expected_share": f"{summary['expected_share']:.0%}",
        "data_checks": checks,
        "measures_compared": WORDS[n] if n < len(WORDS) else str(n),
        "measures_outside_past_range": ", ".join(outside) if outside else "none",
        "same_quarter": rows,
    }
    return {"facts": facts}


def get_docs(state):
    return {"passages": search(state["question"], n=3)}


def _prompt(state):
    parts = [f"QUESTION: {state['question']}"]
    if state.get("facts"):
        parts.append("FACTS:\n" + json.dumps(state["facts"], indent=1))
    if state.get("passages"):
        parts.append("PASSAGES:\n" + "\n\n".join(
            f"[{p['source']}: {p['heading']}]\n{p['text']}" for p in state["passages"]))
    return "\n\n".join(parts)


def _text_of(content_blocks):
    """Join the text blocks of a Claude reply, skipping any reasoning blocks."""
    return "".join(b.text for b in content_blocks if getattr(b, "type", None) == "text")


def call_claude(system, user):
    import anthropic  # reads ANTHROPIC_API_KEY from the environment
    client = anthropic.Anthropic()
    resp = client.messages.create(model=MODEL, max_tokens=4000, system=system,
                                  messages=[{"role": "user", "content": user}])
    if resp.stop_reason == "max_tokens":
        print("warning: answer was cut off by the token limit")
    return _text_of(resp.content)


def _tokens(text):
    return {m.group(0).replace(",", "").rstrip(".") for m in NUM_RE.finditer(text)}


def _fallback(state):
    facts = state.get("facts", {})
    lines = ["I could not check every number in my answer against the results, "
             "so here are the results directly:"]
    lines += [f"- {k}: {v}" for k, v in facts.items() if k != "same_quarter"]
    lines += ["- " + row for row in facts.get("same_quarter", [])]
    return "\n".join(lines)


def after_route(state):
    return "numbers" if state["need_numbers"] else "docs"


def after_numbers(state):
    return "docs" if state["need_docs"] else "answer"


def build_graph(answer_fn=call_claude):
    def node_answer(state):
        return {"answer": answer_fn(SYSTEM_PROMPT, _prompt(state))}

    def node_check(state):
        context = json.dumps(state.get("facts", {})) + " ".join(
            p["text"] for p in state.get("passages", []))
        allowed = _tokens(context) | {"1", "2", "3", "4"}
        unknown = sorted(t for t in _tokens(state["answer"])
                         if t not in allowed and not re.fullmatch(r"(19|20)\d\d", t))
        if not unknown:
            return {"grounded": True, "unknown_numbers": []}
        return {"grounded": False, "unknown_numbers": unknown, "answer": _fallback(state)}
    
    g = StateGraph(AgentState)
    g.add_node("node_route", route)
    g.add_node("node_numbers", get_numbers)
    g.add_node("node_docs", get_docs)
    g.add_node("node_answer", node_answer)
    g.add_node("node_check", node_check)
    g.set_entry_point("node_route")
    g.add_conditional_edges("node_route", after_route,
                            {"numbers": "node_numbers", "docs": "node_docs"})
    g.add_conditional_edges("node_numbers", after_numbers,
                            {"docs": "node_docs", "answer": "node_answer"})
    g.add_edge("node_docs", "node_answer")
    g.add_edge("node_answer", "node_check")
    g.add_edge("node_check", END)
    return g.compile()