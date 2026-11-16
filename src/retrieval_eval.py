"""Evaluate retrieval methods and context packing on dev/test splits."""
import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.retrievers import (
    Bm25Retriever,
    DensePgVectorRetriever,
    HybridRrfRetriever,
    RerankRetriever,
    TfidfRetriever,
    infer_year_filter,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def is_gold_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    c_text = _norm(chunk.get("text", ""))
    return any(
        chunk.get("source_id") == g["source_id"]
        and int(chunk.get("page", -1)) == int(g["page"])
        and _norm(g["evidence"]) in c_text
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget:
            blocks.append(block)
            ids.append(cid)
    return "\n\n".join(blocks), ids


def evaluate_retriever(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    auto_filter: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        flt = infer_year_filter(q["question"]) if auto_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=flt, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context, context_ids = pack_context(results, budget=budget)
        marked = [{**c, "hit": is_gold_hit(c, q["gold"])} for c in results]
        ctx_set = set(context_ids)
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "config": name,
                "applied_filter": flt,
                "top_k": top_k,
                "budget_chars": budget,
                "gold": q["gold"],
                "results": marked,
                "context_ids": context_ids,
                "context_chars": len(context),
                "elapsed_ms": round(elapsed_ms, 2),
                "hit1": bool(marked and marked[0]["hit"]),
                "hit3": any(c["hit"] for c in marked[:3]),
                "context_hit": any(c["hit"] and c["id"] in ctx_set for c in marked),
            }
        )

    n = len(traces)
    h1 = sum(t["hit1"] for t in traces)
    h3 = sum(t["hit3"] for t in traces)
    ctx_h = sum(t["context_hit"] for t in traces)
    cost = 0.0
    if hasattr(retriever, "estimated_cost_usd"):
        cost += float(retriever.estimated_cost_usd)
    if hasattr(retriever, "dense") and hasattr(retriever.dense, "estimated_cost_usd"):
        cost += float(retriever.dense.estimated_cost_usd)
    if hasattr(retriever, "base_retriever") and hasattr(retriever.base_retriever, "dense"):
        cost += float(retriever.base_retriever.dense.estimated_cost_usd)

    summary = {
        "config": name,
        "split": questions[0]["split"] if questions else "",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "auto_filter": auto_filter,
        "hit1": h1,
        "hit1_ratio": f"{h1}/{n}",
        "hit1_pct": round(100.0 * h1 / n, 1) if n else 0.0,
        "hit3": h3,
        "hit3_ratio": f"{h3}/{n}",
        "hit3_pct": round(100.0 * h3 / n, 1) if n else 0.0,
        "context_evidence": ctx_h,
        "context_evidence_ratio": f"{ctx_h}/{n}",
        "context_evidence_pct": round(100.0 * ctx_h / n, 1) if n else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if traces else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2) if traces else 0.0,
        "service_cost_usd": round(cost, 6),
    }
    return traces, summary


def build_retriever(method: str, chunks: list[dict[str, Any]]) -> Any:
    if method == "tfidf":
        return TfidfRetriever(chunks)
    if method == "bm25":
        return Bm25Retriever(chunks)
    settings = get_settings()
    dense = DensePgVectorRetriever(settings)
    if method == "dense":
        return dense
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRrfRetriever(dense=dense, bm25=bm25)
    if method == "hybrid":
        return hybrid
    if method == "rerank":
        return RerankRetriever(base_retriever=hybrid, settings=settings, candidate_k=10)
    raise ValueError(f"Unknown retrieval method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on dev/test")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "starter" / "data" / "converted" / "chunks.json",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "starter" / "data" / "converted" / "questions.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "tfidf:tfidf:3:800:false",
            "dense:dense:3:800:false",
            "hybrid:hybrid:3:800:false",
            "hybrid_imp1:hybrid:5:1800:true",
            "rerank_imp2:rerank:5:1800:true",
        ],
        help="List of name:method:top_k:budget:auto_filter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "retrieval",
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in all_questions if q.get("split") == args.split]

    all_summaries = []
    for spec in args.configs:
        name, method, top_k_s, budget_s, auto_s = spec.split(":")
        top_k = int(top_k_s)
        budget = int(budget_s)
        auto_filter = auto_s.lower() == "true"
        retriever = build_retriever(method, chunks)
        traces, summary = evaluate_retriever(
            name, retriever, selected, top_k=top_k, budget=budget, auto_filter=auto_filter
        )
        out_dir = args.output / f"{name}_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_table_path = args.output / f"summary_{args.split}.json"
    args.output.mkdir(parents=True, exist_ok=True)
    summary_table_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
