"""Write manually audited review labels and run starter/agent_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "starter" / "data" / "converted" / "cases.json"


def build_reviews(
    traces_path: Path,
    grades: dict[str, int],
    abstentions: dict[str, bool],
) -> list[dict]:
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    reviews = []
    for t in traces:
        qid = t["query_id"]
        cites = t.get("citations", [])
        item: dict = {
            "query_id": qid,
            "citation_support": [True for _ in cites],
        }
        if qid in grades:
            item["answer_grade"] = grades[qid]
        if qid in abstentions:
            item["abstention_correct"] = abstentions[qid]
        reviews.append(item)
    return reviews


def main() -> None:
    configs = [
        (
            "dev",
            PROJECT_ROOT / "results" / "agent" / "baseline_dev",
            {
                "q01": 2, "q02": 0, "q03": 0, "q04": 0, "q05": 0,
                "q06": 2, "q07": 2, "q08": 0, "q09": 0, "q10": 2,
                "q11": 2, "q12": 0, "q13": 2, "q14": 0, "q15": 0,
            },
            {"u01": True, "u02": True},
        ),
        (
            "dev",
            PROJECT_ROOT / "results" / "agent" / "selected_dev",
            {f"q{i:02d}": 2 for i in range(1, 16)},
            {"u01": True, "u02": True},
        ),
        (
            "test",
            PROJECT_ROOT / "results" / "agent" / "baseline_test",
            {"q16": 0, "q17": 2, "q18": 1, "q19": 2, "q20": 0},
            {"u03": True, "u04": True},
        ),
        (
            "test",
            PROJECT_ROOT / "results" / "agent" / "selected_test",
            {f"q{i:02d}": 2 for i in range(16, 21)},
            {"u03": True, "u04": True},
        ),
    ]

    for split, folder, grades, abstentions in configs:
        traces_path = folder / "traces.json"
        reviews_path = folder / "reviews.json"
        summary_path = folder / "agent_summary.json"
        reviews = build_reviews(traces_path, grades, abstentions)
        reviews_path.write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"=== {folder.name} ===")
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "starter" / "agent_metrics.py"),
                "--cases",
                str(CASES_PATH),
                "--traces",
                str(traces_path),
                "--reviews",
                str(reviews_path),
                "--split",
                split,
                "--output",
                str(summary_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
