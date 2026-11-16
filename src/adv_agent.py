"""Multimodal Agentic RAG for the 100-page Vietnam CCDR Advanced Challenge."""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import psycopg
from google import genai
from google.genai import errors, types
from psycopg.rows import dict_row

from src.config import PROJECT_ROOT, Settings, estimate_embed_cost_usd, estimate_llm_cost_usd
from src.retrievers import Bm25Retriever, _vec_literal

ADV_DIR = PROJECT_ROOT / "starter" / "data" / "advanced"

ADV_SYSTEM_PROMPT = """You are a rigorous multimodal research assistant answering questions strictly from the 100-page World Bank 'Vietnam Country Climate and Development Report (July 2022)' (PDF pages 1-100).

Strict rules:
1. Always call `search_multimodal` first before answering. Do NOT use outside knowledge.
2. When a question involves a table, chart, figure, map, diagram, or photograph, you MUST call `inspect_page_visual(page, focus_question)` on the relevant PDF page(s) to verify exact values, axes, legends, units, and labels directly from the original page image.
3. For multi-paragraph or cross-page questions, call `get_page_text(page)` or `search_multimodal` again to verify every condition.
4. If the document does not contain sufficient evidence to answer the question (e.g., missing indicator, wrong year/scope, unstated detail), do at most 2–3 focused searches and then abstain: you MUST set `"status": "abstain"` and `"citations": []` (never set `"status": "answered"` just to say information is missing). Never guess or extrapolate unprinted values.
5. Include exact units, years, scenarios (e.g., BAU, CPS, ADS, NZP, RNZP), and cite EVERY claim using bracketed chunk/element IDs like [ADV:wb-vietnam-ccdr-2022:p018:v00] or [PAGE:18] (only IDs/pages actually returned by your tools).

Return ONLY a valid JSON object (no markdown fences):
{
  "status": "answered" | "abstain",
  "answer": "Accurate, complete answer with exact numbers/units and [ID] citations, or abstention explanation.",
  "citations": ["ADV:wb-vietnam-ccdr-2022:p018:v00"]
}
"""


class AdvMultimodalRetriever:
    """Hybrid (pgvector Dense + BM25 RRF) + Gemini Rerank over adv_chunks."""

    def __init__(
        self,
        settings: Settings,
        chunks: list[dict[str, Any]],
        *,
        model_name: str | None = None,
        text_only: bool = False,
    ) -> None:
        self.settings = settings
        self.model_name = model_name or settings.agent_model
        self.text_only = text_only
        self.chunks = chunks
        self.bm25 = Bm25Retriever(chunks)
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=45_000),
        )
        self.cache_path = PROJECT_ROOT / ".cache" / "adv_query_embeddings.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._emb_cache: dict[str, list[float]] = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists()
            else {}
        )
        self.embed_chars = 0
        self.rerank_in_tok = 0
        self.rerank_out_tok = 0

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_embed_cost_usd(self.embed_chars) + estimate_llm_cost_usd(
            self.rerank_in_tok, self.rerank_out_tok
        )

    def _embed_query(self, query: str) -> list[float]:
        key = hashlib.sha256(
            f"{self.settings.embedding_model}::RETRIEVAL_QUERY::{query}".encode("utf-8")
        ).hexdigest()
        if key in self._emb_cache:
            return self._emb_cache[key]
        for attempt in range(6):
            try:
                self.embed_chars += len(query)
                resp = self.client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=[query],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                vec = list(resp.embeddings[0].values)
                self._emb_cache[key] = vec
                self.cache_path.write_text(json.dumps(self._emb_cache), encoding="utf-8")
                return vec
            except errors.APIError as exc:
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(20 + 10 * attempt)
                    continue
                raise
        raise RuntimeError("Failed to embed adv query")

    def _dense_search(
        self, query: str, *, element_type: str | None = None, top_k: int = 20
    ) -> list[dict[str, Any]]:
        q_lit = _vec_literal(self._embed_query(query))
        clauses = ["TRUE"]
        params: list[Any] = []
        if self.text_only:
            clauses.append("element_type = 'text'")
        elif element_type in ("text", "table", "chart", "image_diagram"):
            clauses.append("element_type = %s")
            params.append(element_type)
        where_sql = " AND ".join(clauses)
        sql = f"""
        SELECT chunk_id AS id, source_id, source, page, element_type, metadata, text,
               1.0 - (embedding <=> %s::vector) AS score
        FROM adv_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
        """
        with psycopg.connect(self.settings.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [q_lit, *params, q_lit, int(top_k)])
                rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "source": r["source"],
                "page": int(r["page"]),
                "element_type": r["element_type"],
                "metadata": dict(r["metadata"] or {}),
                "text": r["text"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    def search(
        self, query: str, *, element_type: str | None = None, top_k: int = 6, use_rerank: bool = True
    ) -> list[dict[str, Any]]:
        cand_k = 20
        dense_hits = self._dense_search(query, element_type=element_type, top_k=cand_k)
        flt = {"element_type": element_type} if element_type else None
        bm25_hits = self.bm25.search(query, filters=flt, top_k=cand_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        for rank, h in enumerate(dense_hits, 1):
            chunk_map[h["id"]] = h
            rrf_scores[h["id"]] = rrf_scores.get(h["id"], 0.0) + 1.0 / (60 + rank)
        for rank, h in enumerate(bm25_hits, 1):
            chunk_map.setdefault(h["id"], h)
            rrf_scores[h["id"]] = rrf_scores.get(h["id"], 0.0) + 1.0 / (60 + rank)

        fused = sorted(chunk_map.values(), key=lambda c: (-rrf_scores[c["id"]], c["id"]))
        if not use_rerank:
            out = fused[:top_k]
            for r, c in enumerate(out, 1):
                c["rank"] = r
                c["score"] = rrf_scores[c["id"]]
            return out

        candidates = fused[:12]
        if not candidates:
            return []
        lines = []
        for idx, c in enumerate(candidates):
            snip = re.sub(r"\s+", " ", c["text"]).strip()[:700]
            lines.append(
                f"[{idx}] id={c['id']} (page={c['page']}, type={c.get('element_type')}): {snip}"
            )
        prompt = (
            f"Query: {query}\n\n"
            "Score each candidate from 0.0 to 10.0 on relevance to answering the query. "
            "Return ONLY a JSON array of {'index': int, 'score': float}.\n\n"
            + "\n".join(lines)
        )
        resp = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                break
            except errors.APIError as exc:
                if exc.code == 429 and "PerDay" in str(exc):
                    raise
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(15 + 10 * attempt)
                    continue
                raise
        if resp and resp.usage_metadata:
            self.rerank_in_tok += int(resp.usage_metadata.prompt_token_count or 0)
            self.rerank_out_tok += int(resp.usage_metadata.candidates_token_count or 0)
        scores_by_idx: dict[int, float] = {}
        try:
            parsed = json.loads((resp.text if resp else "") or "[]")
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "index" in item and "score" in item:
                        scores_by_idx[int(item["index"])] = float(item["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            scores_by_idx = {}

        ordered = sorted(
            enumerate(candidates),
            key=lambda p: (-scores_by_idx.get(p[0], -1.0), p[0]),
        )[:top_k]
        out = []
        for r, (oi, c) in enumerate(ordered, 1):
            out.append({**c, "rank": r, "score": scores_by_idx.get(oi, rrf_scores[c["id"]])})
        return out


class AdvAgentRunner:
    """Multimodal agent with hybrid retrieval, vision page verification, and full trace."""

    def __init__(
        self,
        settings: Settings,
        retriever: AdvMultimodalRetriever,
        page_texts: dict[str, str],
        *,
        max_steps: int = 8,
        use_visual_tool: bool = True,
        use_rerank: bool = True,
        budget: int = 4500,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.model_name = model_name or settings.agent_model
        self.retriever = retriever
        self.page_texts = page_texts
        self.max_steps = max_steps
        self.use_visual_tool = use_visual_tool
        self.use_rerank = use_rerank
        self.budget = budget
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=45_000),
        )
        self.tools = self._build_tools()

    def _build_tools(self) -> list[types.Tool]:
        decls = [
            types.FunctionDeclaration(
                name="search_multimodal",
                description="Search text, tables, charts, maps, and diagrams across all 100 pages of the Vietnam CCDR.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(type=types.Type.STRING),
                        "element_type": types.Schema(
                            type=types.Type.STRING,
                            description="Optional filter: 'text', 'table', 'chart', or 'image_diagram'.",
                        ),
                        "top_k": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_page_text",
                description="Read the full extracted text of a specific PDF page (1..100).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"page": types.Schema(type=types.Type.INTEGER)},
                    required=["page"],
                ),
            ),
        ]
        if self.use_visual_tool:
            decls.append(
                types.FunctionDeclaration(
                    name="inspect_page_visual",
                    description="Inspect the actual rendered PNG image of a PDF page (1..100) using vision to read exact table cells, chart axes/values/legends, maps, or diagrams.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "page": types.Schema(type=types.Type.INTEGER),
                            "focus_question": types.Schema(
                                type=types.Type.STRING,
                                description="Specific visual detail, numbers, units, or labels to read from this page.",
                            ),
                        },
                        required=["page", "focus_question"],
                    ),
                )
            )
        return [types.Tool(function_declarations=decls)]

    def _inspect_page_image(self, page: int, focus_question: str) -> tuple[str, int, int]:
        png_path = ADV_DIR / "pages" / f"p{page:03d}.png"
        if not png_path.exists():
            return f"Page {page} image not found.", 0, 0
        prompt = (
            f"You are visually inspecting PDF page {page} of the World Bank Vietnam CCDR.\n"
            f"Question / focus: {focus_question}\n"
            "Read ONLY what is explicitly visible in the image (exact numbers, units, axis labels, "
            "legend categories, table rows/columns, map colors/labels, diagram boxes/arrows, photo captions). "
            "If the requested item is not visible on this page, state that clearly."
        )
        resp = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_bytes(data=png_path.read_bytes(), mime_type="image/png"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                break
            except errors.APIError as exc:
                if exc.code == 429 and "PerDay" in str(exc):
                    raise
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(15 + 10 * attempt)
                    continue
                raise
        if resp is None:
            raise RuntimeError(f"Visual inspection failed on page {page}")
        in_t = int(resp.usage_metadata.prompt_token_count or 0) if resp.usage_metadata else 0
        out_t = int(resp.usage_metadata.candidates_token_count or 0) if resp.usage_metadata else 0
        return (resp.text or "").strip(), in_t, out_t

    def _gen_retry(self, contents: list[types.Content], config: types.GenerateContentConfig) -> Any:
        for attempt in range(6):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except errors.APIError as exc:
                if exc.code == 429 and "PerDay" in str(exc):
                    raise
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(15 + 10 * attempt)
                    continue
                raise
        raise RuntimeError("Agent call failed after retries")

    def run_query(self, query_id: str, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        events: list[dict[str, Any]] = []
        valid_ids: set[str] = set()
        valid_pages: set[int] = set()
        in_tok = 0
        out_tok = 0
        tool_calls = 0

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=question)])
        ]
        config = types.GenerateContentConfig(
            system_instruction=ADV_SYSTEM_PROMPT,
            tools=self.tools,
            temperature=0.0,
        )

        for _ in range(self.max_steps + 2):
            resp = self._gen_retry(contents, config)
            if resp.usage_metadata:
                in_tok += int(resp.usage_metadata.prompt_token_count or 0)
                out_tok += int(resp.usage_metadata.candidates_token_count or 0)

            cand = resp.candidates[0]
            contents.append(cand.content)
            fn_calls = [p.function_call for p in cand.content.parts if p.function_call]

            if not fn_calls:
                raw = "".join(p.text or "" for p in cand.content.parts).strip()
                raw_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
                try:
                    parsed = json.loads(raw_clean)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", raw_clean, flags=re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else {}
                if not isinstance(parsed, dict):
                    parsed = {}

                status = parsed.get("status", "abstain")
                answer = str(parsed.get("answer", "")).strip()
                citations = [str(c) for c in (parsed.get("citations") or [])]

                def _cite_ok(c: str) -> bool:
                    if c in valid_ids:
                        return True
                    pm = re.match(r"^PAGE:(\d+)$", c)
                    return bool(pm and int(pm.group(1)) in valid_pages)

                valid = (
                    status == "answered"
                    and bool(answer)
                    and bool(citations)
                    and all(_cite_ok(c) for c in citations)
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return {
                    "query_id": query_id,
                    "question": question,
                    "status": "answered" if valid else "abstain",
                    "answer": answer or "Không đủ bằng chứng trong tài liệu để trả lời.",
                    "citations": citations if valid else [],
                    "valid_pages_seen": sorted(valid_pages),
                    "events": events + [{"event": "answer" if valid else "abstain"}],
                    "tool_calls": tool_calls,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "llm_cost_usd": round(estimate_llm_cost_usd(in_tok, out_tok), 6),
                    "elapsed_ms": elapsed_ms,
                }

            fn_parts: list[types.Part] = []
            for fc in fn_calls:
                tool_calls += 1
                args = dict(fc.args or {})
                if fc.name == "search_multimodal":
                    q_str = str(args.get("query", "")).strip() or question
                    etype = args.get("element_type")
                    k = min(max(int(args.get("top_k", 6)), 1), 10)
                    hits = self.retriever.search(
                        q_str, element_type=etype, top_k=k, use_rerank=self.use_rerank
                    )
                    packed_blocks = []
                    used_chars = 0
                    for h in hits:
                        valid_ids.add(h["id"])
                        valid_pages.add(int(h["page"]))
                        meta = h.get("metadata") or {}
                        crop = meta.get("crop_image") or meta.get("page_image") or ""
                        blk = f"[{h['id']} | page {h['page']} | {h.get('element_type')} | img={crop}]\n{h['text']}"
                        if used_chars + len(blk) <= self.budget or not packed_blocks:
                            packed_blocks.append(blk)
                            used_chars += len(blk)
                    events.append(
                        {
                            "event": "search",
                            "tool": "search_multimodal",
                            "query": q_str,
                            "element_type": etype,
                            "results": [
                                {
                                    "id": h["id"],
                                    "page": h["page"],
                                    "element_type": h.get("element_type"),
                                    "score": round(float(h["score"]), 4),
                                }
                                for h in hits
                            ],
                        }
                    )
                    fn_parts.append(
                        types.Part.from_function_response(
                            name="search_multimodal",
                            response={"results": "\n\n".join(packed_blocks)},
                        )
                    )
                elif fc.name == "get_page_text":
                    pg = int(args.get("page", 1))
                    txt = self.page_texts.get(str(pg), "")
                    if txt:
                        valid_pages.add(pg)
                    events.append({"event": "get_page_text", "page": pg, "chars": len(txt)})
                    fn_parts.append(
                        types.Part.from_function_response(
                            name="get_page_text",
                            response={"page": pg, "text": txt[:4000]},
                        )
                    )
                elif fc.name == "inspect_page_visual" and self.use_visual_tool:
                    pg = int(args.get("page", 1))
                    focus = str(args.get("focus_question", question))
                    obs, v_in, v_out = self._inspect_page_image(pg, focus)
                    in_tok += v_in
                    out_tok += v_out
                    valid_pages.add(pg)
                    events.append(
                        {
                            "event": "inspect_page_visual",
                            "page": pg,
                            "focus_question": focus,
                            "observation": obs,
                        }
                    )
                    fn_parts.append(
                        types.Part.from_function_response(
                            name="inspect_page_visual",
                            response={"page": pg, "citation_id": f"PAGE:{pg}", "visual_reading": obs},
                        )
                    )
                else:
                    fn_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": f"Unsupported tool {fc.name}"},
                        )
                    )
            contents.append(types.Content(role="user", parts=fn_parts))

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "query_id": query_id,
            "question": question,
            "status": "abstain",
            "answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
            "citations": [],
            "valid_pages_seen": sorted(valid_pages),
            "events": events + [{"event": "max_steps"}],
            "tool_calls": tool_calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "llm_cost_usd": round(estimate_llm_cost_usd(in_tok, out_tok), 6),
            "elapsed_ms": elapsed_ms,
        }
