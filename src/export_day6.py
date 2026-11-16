"""Export Day 6 chunks from PostgreSQL/pgvector and build Day 7 benchmark files."""
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row

from src.config import PROJECT_ROOT, get_settings

DAY6_QUESTIONS_PATH = (
    PROJECT_ROOT.parent
    / "TH day 06"
    / "K5-L3-Day06-DataIngestionEngineering"
    / "benchmark"
    / "questions.json"
)
OUT_DIR = PROJECT_ROOT / "starter" / "data" / "converted"

UNANSWERABLE_CASES = [
    {
        "id": "u01",
        "split": "dev",
        "question": "What was Vietnam's exact Gini coefficient for income inequality in 2024 according to the Taking Stock reports?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 88 content pages (558 chunks): the term 'Gini' and 2024 income inequality Gini coefficient do not appear in either Taking Stock 08/2023 or 03/2025.",
    },
    {
        "id": "u02",
        "split": "dev",
        "question": "What specific corporate income tax rate did Vietnam's National Assembly approve for semiconductor foreign investors in 2025?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 88 content pages (558 chunks): 'semiconductor' appears in 0 chunks; no semiconductor-specific CIT rate is reported.",
    },
    {
        "id": "u03",
        "split": "test",
        "question": "What was the total passenger volume of Long Thanh International Airport in Q1 2025?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 88 content pages (558 chunks): 'Long Thanh' appears in 0 chunks; airport passenger volume is not covered.",
    },
    {
        "id": "u04",
        "split": "test",
        "question": "What was Vietnam's bilateral trade deficit with South Korea in Q4 2024?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Verified across all 88 content pages (558 chunks): 'South Korea' appears in 0 chunks; bilateral Q4 2024 trade balance with South Korea is not reported.",
    },
]


def export_day6() -> None:
    settings = get_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document_id, source, page, year, section, metadata, text
                FROM chunks
                WHERE strategy = %s
                ORDER BY chunk_id
                """,
                (settings.day6_strategy,),
            )
            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"No chunks found for strategy={settings.day6_strategy}")

    chunks = []
    for r in rows:
        meta = dict(r["metadata"] or {})
        if r["year"] is not None:
            meta.setdefault("year", r["year"])
        if r["section"] is not None:
            meta.setdefault("section", r["section"])
        chunks.append(
            {
                "id": r["chunk_id"],
                "source_id": r["document_id"],
                "source": r["source"],
                "page": int(r["page"]),
                "text": r["text"],
                "metadata": meta,
            }
        )

    raw_questions = json.loads(DAY6_QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = []
    cases = []
    for q in raw_questions:
        pages = q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]]
        gold = [
            {
                "source_id": q["gold_source_id"],
                "page": int(p),
                "evidence": q["evidence_quote"],
            }
            for p in pages
        ]
        item = {
            "id": q["id"],
            "split": q["split"],
            "type": q.get("type", "direct"),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": gold,
            "filter": q.get("filter"),
        }
        questions.append(item)
        cases.append(
            {
                "id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "answerable": True,
                "expected_answer": q["expected_answer"],
                "gold": gold,
                "filter": q.get("filter"),
            }
        )

    cases.extend(UNANSWERABLE_CASES)

    (OUT_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Exported {len(chunks)} chunks (strategy={settings.day6_strategy}), "
        f"{len(questions)} answerable questions, {len(cases)} total cases to {OUT_DIR}"
    )


if __name__ == "__main__":
    export_day6()
