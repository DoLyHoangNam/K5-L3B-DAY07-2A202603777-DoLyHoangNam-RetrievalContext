"""Retrieval methods for Day 7: TF-IDF, BM25, Dense (pgvector), Hybrid (RRF), and LLM Rerank."""
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg
from google import genai
from google.genai import errors, types
from psycopg.rows import dict_row

from src.config import PROJECT_ROOT, Settings, estimate_embed_cost_usd, estimate_llm_cost_usd

ALLOWED_FILTER_KEYS = {"year", "section", "document_id"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text or "").casefold())


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def _matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for k, v in filters.items():
        if k == "document_id":
            if chunk.get("source_id") != v:
                return False
        elif meta.get(k) != v:
            return False
    return True


def infer_year_filter(query: str) -> dict[str, Any] | None:
    """Infer report year filter only from explicit edition mentions in the query."""
    q = query.casefold()
    mentions_2023_edition = bool(
        re.search(r"\b(2023 report|august 2023|08/2023|taking stock 2023)\b", q)
    )
    mentions_2025_edition = bool(
        re.search(r"\b(2025 report|march 2025|03/2025|taking stock 2025)\b", q)
    )
    if mentions_2023_edition and not mentions_2025_edition:
        return {"year": 2023}
    if mentions_2025_edition and not mentions_2023_edition:
        return {"year": 2025}
    return None


class TfidfRetriever:
    """Offline TF-IDF lexical retriever."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        n = len(chunks)
        self.idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: cnt * self.idf[t] for t, cnt in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vec in zip(self.chunks, self.vectors):
            if not _matches_filter(chunk, filters):
                continue
            score = sum(val * vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, "score": float(score), "strategy": "tfidf"})
        ranked = sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        return ranked


class Bm25Retriever:
    """Okapi BM25 lexical retriever with optional metadata filtering."""

    def __init__(
        self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [Counter(tokenize(c["text"])) for c in chunks]
        self.doc_lens = [sum(cnt.values()) for cnt in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(chunks)) if chunks else 1.0
        df = Counter(term for cnt in self.doc_tokens for term in cnt)
        n = len(chunks)
        self.idf = {
            term: math.log((n - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_terms = tokenize(query)
        results = []
        for chunk, tf_map, dl in zip(self.chunks, self.doc_tokens, self.doc_lens):
            if not _matches_filter(chunk, filters):
                continue
            score = 0.0
            for term in q_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0:
                results.append({**chunk, "score": float(score), "strategy": "bm25"})
        ranked = sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        return ranked


class DensePgVectorRetriever:
    """Dense retriever backed by Day 6 PostgreSQL + pgvector and Gemini embeddings."""

    def __init__(self, settings: Settings, cache_path: Path | None = None) -> None:
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=30_000),
        )
        self.cache_path = cache_path or (PROJECT_ROOT / ".cache" / "query_embeddings.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            self._cache: dict[str, list[float]] = json.loads(
                self.cache_path.read_text(encoding="utf-8")
            )
        else:
            self._cache = {}
        self.api_calls = 0
        self.embedded_chars = 0

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_embed_cost_usd(self.embedded_chars)

    def embed_query(self, query: str) -> list[float]:
        key = hashlib.sha256(
            f"{self.settings.embedding_model}::RETRIEVAL_QUERY::{query}".encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]

        for attempt in range(5):
            try:
                self.api_calls += 1
                self.embedded_chars += len(query)
                resp = self.client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=[query],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                vec = list(resp.embeddings[0].values)
                self._cache[key] = vec
                self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
                return vec
            except errors.ClientError as exc:
                if exc.code == 429 and attempt < 4:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Failed to embed query after retries")

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_vec = self.embed_query(query)
        q_lit = _vec_literal(q_vec)
        clauses = ["strategy = %s"]
        params: list[Any] = [self.settings.day6_strategy]
        if filters:
            for k, v in filters.items():
                if k not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {k}")
                clauses.append(f"{k} = %s")
                params.append(v)

        where_sql = " AND ".join(clauses)
        sql = f"""
        SELECT
            chunk_id AS id,
            document_id AS source_id,
            source,
            page,
            year,
            section,
            metadata,
            text,
            1.0 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
        """
        exec_params = [q_lit, *params, q_lit, int(top_k)]
        with psycopg.connect(self.settings.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, exec_params)
                rows = cur.fetchall()

        results = []
        for rank, r in enumerate(rows, 1):
            meta = dict(r["metadata"] or {})
            if r["year"] is not None:
                meta.setdefault("year", r["year"])
            if r["section"] is not None:
                meta.setdefault("section", r["section"])
            results.append(
                {
                    "id": r["id"],
                    "source_id": r["source_id"],
                    "source": r["source"],
                    "page": int(r["page"]),
                    "metadata": meta,
                    "text": r["text"],
                    "score": float(r["score"]),
                    "rank": rank,
                    "strategy": "dense_pgvector",
                }
            )
        return results


class HybridRrfRetriever:
    """Hybrid retriever fusing Dense (pgvector) and BM25 via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DensePgVectorRetriever,
        bm25: Bm25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 15,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        k_cand = max(self.candidate_k, top_k * 3)
        dense_hits = self.dense.search(query, filters=filters, top_k=k_cand)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=k_cand)

        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        for rank, h in enumerate(dense_hits, 1):
            cid = h["id"]
            chunk_map[cid] = h
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, h in enumerate(bm25_hits, 1):
            cid = h["id"]
            chunk_map.setdefault(cid, h)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(
            chunk_map.values(),
            key=lambda c: (-fused_scores[c["id"]], c["id"]),
        )[:top_k]
        results = []
        for rank, c in enumerate(ordered, 1):
            results.append(
                {
                    **c,
                    "score": float(fused_scores[c["id"]]),
                    "rank": rank,
                    "strategy": "hybrid_rrf",
                }
            )
        return results


class RerankRetriever:
    """Two-stage retriever: Hybrid RRF candidates -> Gemini Flash relevance reranking."""

    def __init__(
        self,
        base_retriever: HybridRrfRetriever,
        settings: Settings,
        candidate_k: int = 10,
    ) -> None:
        self.base_retriever = base_retriever
        self.settings = settings
        self.candidate_k = candidate_k
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=30_000),
        )
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_llm_cost_usd(self.input_tokens, self.output_tokens)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        candidates = self.base_retriever.search(
            query, filters=filters, top_k=max(self.candidate_k, top_k)
        )
        if not candidates:
            return []

        cand_lines = []
        for idx, c in enumerate(candidates):
            snippet = re.sub(r"\s+", " ", c["text"]).strip()
            cand_lines.append(
                f"[{idx}] id={c['id']} (source={c['source_id']}, page={c['page']}): {snippet}"
            )

        prompt = (
            "You are a relevance reranker for economic document retrieval.\n"
            f"Query: {query}\n\n"
            "Score each candidate chunk from 0.0 to 10.0 for how directly it contains "
            "evidence to answer the query. Return ONLY a JSON array of objects with "
            "keys 'index' (int) and 'score' (float), ordered from highest to lowest score.\n\n"
            "Candidates:\n" + "\n".join(cand_lines)
        )

        resp = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.settings.agent_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                break
            except errors.APIError as exc:
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(20 + 10 * attempt)
                    continue
                raise
        if resp is None:
            raise RuntimeError("Reranker failed after retries")
        self.api_calls += 1
        if resp.usage_metadata:
            self.input_tokens += int(resp.usage_metadata.prompt_token_count or 0)
            self.output_tokens += int(resp.usage_metadata.candidates_token_count or 0)

        scores_by_idx: dict[int, float] = {}
        try:
            parsed = json.loads(resp.text or "[]")
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "index" in item and "score" in item:
                        scores_by_idx[int(item["index"])] = float(item["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            scores_by_idx = {}

        ranked = sorted(
            enumerate(candidates),
            key=lambda pair: (-scores_by_idx.get(pair[0], -1.0), pair[0]),
        )[:top_k]

        results = []
        for rank, (orig_idx, c) in enumerate(ranked, 1):
            rr_score = scores_by_idx.get(orig_idx, c["score"])
            results.append(
                {
                    **c,
                    "score": float(rr_score),
                    "rank": rank,
                    "strategy": "hybrid_rerank",
                }
            )
        return results


class AdjacentChunkStore:
    """Lookup adjacent chunks on the same document/page for context expansion."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.by_id = {c["id"]: c for c in chunks}
        self.ordered_by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            self.ordered_by_doc.setdefault(c["source_id"], []).append(c)
        for doc_id in self.ordered_by_doc:
            self.ordered_by_doc[doc_id].sort(key=lambda x: x["id"])
        self.pos_by_id: dict[str, tuple[str, int]] = {}
        for doc_id, doc_chunks in self.ordered_by_doc.items():
            for idx, c in enumerate(doc_chunks):
                self.pos_by_id[c["id"]] = (doc_id, idx)

    def get_adjacent(self, chunk_id: str, window: int = 1) -> list[dict[str, Any]]:
        if chunk_id not in self.pos_by_id:
            return []
        doc_id, idx = self.pos_by_id[chunk_id]
        doc_chunks = self.ordered_by_doc[doc_id]
        lo = max(0, idx - window)
        hi = min(len(doc_chunks), idx + window + 1)
        return [doc_chunks[i] for i in range(lo, hi) if i != idx]
