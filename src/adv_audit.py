"""Inspect Advanced Challenge traces alongside expected answers for manual verification."""
import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "starter" / "data" / "advanced" / "adv_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=Path, required=True)
    args = parser.parse_args()

    cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))}
    traces = json.loads(args.trace_file.read_text(encoding="utf-8"))
    for t in traces:
        qid = t["query_id"]
        c = cases[qid]
        print(f"=== {qid} ({c['group']}) | answerable={c['answerable']} | status={t['status']} ===")
        print(f"Q: {c['question']}")
        if t.get("error"):
            print(f"ERROR: {t['error']}")
        if c["answerable"]:
            print(f"EXP: {c['expected_answer']}")
            print(f"GOLD_EV: {c['evidence']}")
        print(f"GOT: {t['answer']}")
        print(f"CITES: {t['citations']} | PAGES_SEEN: {t.get('valid_pages_seen')}")
        tools = [(e.get("event"), e.get("page") or e.get("query")) for e in t.get("events", [])]
        print(f"TOOLS: {tools}")
        print()


if __name__ == "__main__":
    main()
