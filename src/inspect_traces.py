"""Print agent traces alongside expected answers and cited context snippets for manual audit."""
import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        c["id"]: c
        for c in json.loads(
            (PROJECT_ROOT / "starter" / "data" / "converted" / "cases.json").read_text(
                encoding="utf-8"
            )
        )
    }
    traces = json.loads(args.trace_file.read_text(encoding="utf-8"))
    for t in traces:
        c = cases[t["query_id"]]
        print(f"=== {t['query_id']} | answerable={c['answerable']} | status={t['status']} ===")
        print(f"Q: {c['question']}")
        if c["answerable"]:
            print(f"EXP: {c['expected_answer']}")
        print(f"GOT: {t['answer']}")
        print(f"CITES: {t['citations']}")
        for cid in t["citations"]:
            idx = t["context"].find("[" + cid)
            snippet = t["context"][idx : idx + 350].replace("\n", " ") if idx >= 0 else "NOT IN CTX"
            print(f"  CTX[{cid}]: {snippet}")
        print()


if __name__ == "__main__":
    main()
