"""Run Advanced Challenge multimodal agent evaluation on dev or test split."""
import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.adv_agent import AdvAgentRunner, AdvMultimodalRetriever
from src.config import PROJECT_ROOT, get_settings

ADV_DIR = PROJECT_ROOT / "starter" / "data" / "advanced"


def run_split(
    config_name: str,
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    page_texts: dict[str, str],
    *,
    use_visual_tool: bool,
    use_rerank: bool,
    budget: int,
    out_dir: Path,
    model_name: str = "gemini-3.5-flash",
    text_only: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    retriever = AdvMultimodalRetriever(
        settings=settings, chunks=chunks, model_name=model_name, text_only=text_only
    )
    runner = AdvAgentRunner(
        settings=settings,
        retriever=retriever,
        page_texts=page_texts,
        max_steps=8,
        use_visual_tool=use_visual_tool,
        use_rerank=use_rerank,
        budget=budget,
        model_name=model_name,
    )

    traces: list[dict[str, Any]] = []
    for case in cases:
        cid = case["id"]
        try:
            tr = runner.run_query(cid, case["question"])
        except Exception as exc:
            tr = {
                "query_id": cid,
                "question": case["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "valid_pages_seen": [],
                "events": [],
                "tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_cost_usd": 0.0,
                "elapsed_ms": 0.0,
            }
        tr["group"] = case["group"]
        tr["answerable"] = case["answerable"]
        traces.append(tr)
        print(
            f"[{config_name}] {cid} ({case['group']}) -> status={tr['status']} "
            f"tools={tr['tool_calls']} cites={tr['citations']} ms={tr['elapsed_ms']}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ok = [t for t in traces if t["status"] != "error"]
    summary = {
        "config": config_name,
        "use_visual_tool": use_visual_tool,
        "use_rerank": use_rerank,
        "budget": budget,
        "total": len(cases),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstained": sum(1 for t in traces if t["status"] == "abstain"),
        "avg_tool_calls": round(statistics.mean(t["tool_calls"] for t in ok), 2) if ok else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in ok), 1) if ok else 0.0,
        "agent_llm_cost_usd": round(sum(t["llm_cost_usd"] for t in traces), 6),
        "retriever_cost_usd": round(retriever.estimated_cost_usd, 6),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--config",
        choices=("text_baseline", "multimodal_vision", "both"),
        default="multimodal_vision",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
    )
    args = parser.parse_args()

    chunks = json.loads((ADV_DIR / "adv_chunks.json").read_text(encoding="utf-8"))
    page_texts = json.loads((ADV_DIR / "page_texts.json").read_text(encoding="utf-8"))
    all_cases = json.loads((ADV_DIR / "adv_cases.json").read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c.get("split") == args.split]

    out_root = PROJECT_ROOT / "results" / "advanced"
    if args.config in ("text_baseline", "both"):
        text_only_chunks = [c for c in chunks if c["element_type"] == "text"]
        run_split(
            "adv_text_baseline",
            cases,
            text_only_chunks,
            page_texts,
            use_visual_tool=False,
            use_rerank=False,
            budget=2500,
            out_dir=out_root / f"text_baseline_{args.split}",
            model_name=args.model,
            text_only=True,
        )
    if args.config in ("multimodal_vision", "both"):
        run_split(
            "adv_multimodal_vision",
            cases,
            chunks,
            page_texts,
            use_visual_tool=True,
            use_rerank=args.use_rerank,
            budget=4500,
            out_dir=out_root / f"multimodal_vision_{args.split}",
            model_name=args.model,
            text_only=False,
        )


if __name__ == "__main__":
    main()
