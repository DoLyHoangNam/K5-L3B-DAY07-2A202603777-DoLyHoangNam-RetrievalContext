"""Run real Agentic RAG evaluation on dev or test split and save full traces."""
import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.agent import AgenticRagRunner
from src.config import PROJECT_ROOT, get_settings
from src.retrieval_eval import build_retriever
from src.retrievers import AdjacentChunkStore


def run_agent_config(
    config_name: str,
    method: str,
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    max_searches: int,
    top_k: int,
    budget: int,
    enable_adjacent: bool,
    out_dir: Path,
) -> dict[str, Any]:
    settings = get_settings()
    retriever = build_retriever(method, chunks)
    adjacent_store = AdjacentChunkStore(chunks) if enable_adjacent else None
    runner = AgenticRagRunner(
        settings=settings,
        retriever=retriever,
        adjacent_store=adjacent_store,
        max_searches=max_searches,
        default_top_k=top_k,
        budget=budget,
        enable_adjacent=enable_adjacent,
    )

    traces: list[dict[str, Any]] = []
    for case in cases:
        cid = case["id"]
        try:
            trace = runner.run_query(cid, case["question"])
        except Exception as exc:
            trace = {
                "query_id": cid,
                "question": case["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "context": "",
                "context_ids": [],
                "events": [],
                "search_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_cost_usd": 0.0,
                "elapsed_ms": 0.0,
            }
        traces.append(trace)
        print(
            f"[{config_name}] {cid} -> status={trace['status']} "
            f"searches={trace['search_calls']} citations={trace['citations']} "
            f"ms={trace['elapsed_ms']}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ret_cost = 0.0
    if hasattr(retriever, "estimated_cost_usd"):
        ret_cost += float(retriever.estimated_cost_usd)
    if hasattr(retriever, "dense") and hasattr(retriever.dense, "estimated_cost_usd"):
        ret_cost += float(retriever.dense.estimated_cost_usd)
    if hasattr(retriever, "base_retriever") and hasattr(retriever.base_retriever, "dense"):
        ret_cost += float(retriever.base_retriever.dense.estimated_cost_usd)

    ok_traces = [t for t in traces if t["status"] != "error"]
    summary = {
        "config": config_name,
        "method": method,
        "max_searches": max_searches,
        "top_k": top_k,
        "budget_chars": budget,
        "enable_adjacent": enable_adjacent,
        "total_cases": len(cases),
        "errors": sum(1 for t in traces if t["status"] == "error"),
        "answered": sum(1 for t in traces if t["status"] == "answered"),
        "abstained": sum(1 for t in traces if t["status"] == "abstain"),
        "avg_search_calls": round(
            statistics.mean(t["search_calls"] for t in ok_traces), 2
        )
        if ok_traces
        else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in ok_traces), 1)
        if ok_traces
        else 0.0,
        "total_llm_cost_usd": round(sum(t["llm_cost_usd"] for t in traces), 6),
        "retriever_cost_usd": round(ret_cost, 6),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic RAG on dev or test split")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "starter" / "data" / "converted" / "chunks.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "starter" / "data" / "converted" / "cases.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--config",
        choices=("baseline", "selected", "both"),
        default="both",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "agent",
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [c for c in all_cases if c.get("split") == args.split]

    if args.config in ("baseline", "both"):
        run_agent_config(
            "agent_baseline",
            "dense",
            cases,
            chunks,
            max_searches=2,
            top_k=3,
            budget=800,
            enable_adjacent=False,
            out_dir=args.output / f"baseline_{args.split}",
        )
    if args.config in ("selected", "both"):
        run_agent_config(
            "agent_selected",
            "rerank",
            cases,
            chunks,
            max_searches=4,
            top_k=5,
            budget=2200,
            enable_adjacent=True,
            out_dir=args.output / f"selected_{args.split}",
        )


if __name__ == "__main__":
    main()
