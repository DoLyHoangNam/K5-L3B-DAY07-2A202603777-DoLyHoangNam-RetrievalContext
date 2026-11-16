"""Full-document multimodal ingestion for the 100-page Vietnam CCDR PDF."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pypdf
import pypdfium2 as pdfium
import psycopg
from google import genai
from google.genai import errors, types
from PIL import Image

from src.config import PROJECT_ROOT, Settings, estimate_embed_cost_usd, estimate_llm_cost_usd, get_settings

ADV_DIR = PROJECT_ROOT / "starter" / "data" / "advanced"
PDF_PATH = ADV_DIR / "vietnam_ccdr_full.pdf"
PAGES_DIR = ADV_DIR / "pages"
CROPS_DIR = ADV_DIR / "crops"
DOC_ID = "wb-vietnam-ccdr-2022"
DOC_SOURCE = "starter/data/advanced/vietnam_ccdr_full.pdf"
DOC_URL = "https://documents1.worldbank.org/curated/en/099355507062229876/pdf/P17724103036ee07909701028a237b6d18b.pdf"


def chunk_page_text(text: str, page: int, size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) < 40:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        piece = cleaned[start:end].strip()
        if len(piece) >= 30:
            chunks.append(
                {
                    "id": f"ADV:{DOC_ID}:p{page:03d}:t{idx:02d}",
                    "source_id": DOC_ID,
                    "source": DOC_SOURCE,
                    "page": page,
                    "element_type": "text",
                    "text": piece,
                    "metadata": {
                        "page": page,
                        "element_type": "text",
                        "page_image": f"starter/data/advanced/pages/p{page:03d}.png",
                    },
                }
            )
            idx += 1
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_visual_elements_for_page(
    client: genai.Client,
    model_name: str,
    page: int,
    page_png: Path,
    page_text: str,
    cache: dict[str, Any],
    cache_path: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    key = f"{model_name}::p{page:03d}"
    if key in cache:
        return cache[key], 0, 0

    img_bytes = page_png.read_bytes()
    prompt = (
        f"You are indexing page {page} of the World Bank 'Vietnam Country Climate and Development Report (2022)'.\n"
        "Inspect the page image carefully. Identify ONLY meaningful analytical visual elements on this page:\n"
        "- 'table': data tables, priority matrices, or annex statistical tables.\n"
        "- 'chart': bar charts, line charts, scatter plots, stacked charts, pie/donut charts.\n"
        "- 'image_diagram': conceptual diagrams, flowcharts, maps with legends/data, or informative photographs with captions.\n"
        "Ignore decorative icons, page headers/footers, and plain body text paragraphs.\n"
        "For EACH visual element found, return a JSON object with:\n"
        "  - 'kind': 'table' | 'chart' | 'image_diagram'\n"
        "  - 'title': exact caption/heading on the page (e.g. 'Table 1. Cross-Country Emission Overview', 'Figure 3: Estimated costs...')\n"
        "  - 'bbox_pct': [ymin, xmin, ymax, xmax] integers in 0..1000 bounding the visual element and its title/legend.\n"
        "  - 'summary': comprehensive faithful extraction: for tables include markdown table with ALL headers, rows, and units; "
        "for charts include chart type, axis labels, units, legend series, exact visible numbers/trends; "
        "for diagrams/maps/photos include all labels, regions, color legend meanings, and key facts.\n"
        "If there are NO analytical tables, charts, maps, diagrams, or informative photos on this page, return [].\n"
        "Return ONLY a JSON array.\n\n"
        f"Page {page} PDF text excerpt (for spelling reference):\n{page_text[:1500]}"
    )

    resp = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
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
        raise RuntimeError(f"Vision extraction failed on page {page}")

    in_tok = int(resp.usage_metadata.prompt_token_count or 0) if resp.usage_metadata else 0
    out_tok = int(resp.usage_metadata.candidates_token_count or 0) if resp.usage_metadata else 0

    elements: list[dict[str, Any]] = []
    try:
        parsed = json.loads(resp.text or "[]")
        if isinstance(parsed, list):
            for el in parsed:
                if isinstance(el, dict) and el.get("kind") in ("table", "chart", "image_diagram"):
                    elements.append(el)
    except (json.JSONDecodeError, ValueError, TypeError):
        elements = []

    cache[key] = elements
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return elements, in_tok, out_tok


def crop_element(page_png: Path, bbox_pct: list[int] | None, out_path: Path) -> str:
    if not bbox_pct or len(bbox_pct) != 4:
        return f"starter/data/advanced/pages/{page_png.name}"
    try:
        ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in bbox_pct]
        if ymax <= ymin + 20 or xmax <= xmin + 20:
            return f"starter/data/advanced/pages/{page_png.name}"
        with Image.open(page_png) as im:
            w, h = im.size
            box = (
                int((xmin / 1000.0) * w),
                int((ymin / 1000.0) * h),
                int((xmax / 1000.0) * w),
                int((ymax / 1000.0) * h),
            )
            cropped = im.crop(box)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out_path)
            return f"starter/data/advanced/crops/{out_path.name}"
    except Exception:
        return f"starter/data/advanced/pages/{page_png.name}"


def embed_all(
    client: genai.Client,
    model_name: str,
    texts: list[str],
    cache_path: Path,
) -> tuple[list[list[float]], int]:
    cache: dict[str, list[float]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    results: list[list[float] | None] = [None] * len(texts)
    uncached: list[tuple[int, str, str]] = []
    for i, t in enumerate(texts):
        k = hashlib.sha256(f"{model_name}::RETRIEVAL_DOCUMENT::{t}".encode("utf-8")).hexdigest()
        if k in cache:
            results[i] = cache[k]
        else:
            uncached.append((i, t, k))

    embedded_chars = 0
    batch_size = 60
    for b in range(0, len(uncached), batch_size):
        batch = uncached[b : b + batch_size]
        batch_texts = [item[1] for item in batch]
        embedded_chars += sum(len(x) for x in batch_texts)
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=model_name,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                for (orig_i, _, k), emb in zip(batch, resp.embeddings):
                    vec = list(emb.values)
                    results[orig_i] = vec
                    cache[k] = vec
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                break
            except errors.APIError as exc:
                if exc.code in (429, 503) and attempt < 5:
                    time.sleep(20 + 10 * attempt)
                    continue
                raise
        print(f"  Embedded batch {min(b + batch_size, len(uncached))}/{len(uncached)}", flush=True)

    return [r for r in results if r is not None], embedded_chars


def store_pgvector(settings: Settings, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    dims = len(vectors[0])
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS adv_chunks (
        chunk_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        source TEXT NOT NULL,
        page INTEGER NOT NULL,
        element_type TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        text TEXT NOT NULL,
        embedding vector({dims}) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_adv_chunks_page_type ON adv_chunks (page, element_type);
    """
    upsert_sql = """
    INSERT INTO adv_chunks (chunk_id, source_id, source, page, element_type, metadata, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
    ON CONFLICT (chunk_id) DO UPDATE SET
        source_id = EXCLUDED.source_id,
        source = EXCLUDED.source,
        page = EXCLUDED.page,
        element_type = EXCLUDED.element_type,
        metadata = EXCLUDED.metadata,
        text = EXCLUDED.text,
        embedding = EXCLUDED.embedding;
    """
    keep_ids = [c["id"] for c in chunks]
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute("DELETE FROM adv_chunks WHERE NOT (chunk_id = ANY(%s))", (keep_ids,))
            rows = [
                (
                    c["id"],
                    c["source_id"],
                    c["source"],
                    c["page"],
                    c["element_type"],
                    json.dumps(c["metadata"], ensure_ascii=False),
                    c["text"],
                    "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]",
                )
                for c, vec in zip(chunks, vectors)
            ]
            cur.executemany(upsert_sql, rows)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index 100-page Vietnam CCDR PDF multimodally")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    settings = get_settings()
    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=45_000),
    )

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(str(PDF_PATH))
    pdf_doc = pdfium.PdfDocument(str(PDF_PATH))
    total_pages = len(reader.pages)
    print(f"Opened {PDF_PATH.name}: {total_pages} physical pages", flush=True)

    all_chunks: list[dict[str, Any]] = []
    page_texts: dict[str, str] = {}
    skipped_pages: list[dict[str, Any]] = []
    visual_candidate_pages: list[int] = []

    for idx in range(total_pages):
        page_num = idx + 1
        page_obj = reader.pages[idx]
        raw_text = page_obj.extract_text() or ""
        cleaned_len = len(re.sub(r"\s+", " ", raw_text).strip())
        try:
            n_imgs = len(page_obj.images)
        except Exception:
            n_imgs = 0

        bitmap = pdf_doc[idx].render(scale=150 / 72)
        pil_img = bitmap.to_pil()
        png_path = PAGES_DIR / f"p{page_num:03d}.png"
        pil_img.save(png_path)

        page_texts[str(page_num)] = re.sub(r"\s+", " ", raw_text).strip()

        if cleaned_len < 30 and n_imgs <= 1:
            skipped_pages.append(
                {
                    "page": page_num,
                    "reason": "Blank/back-cover page with no extractable analytical text or figure",
                    "chars": cleaned_len,
                }
            )
            continue

        t_chunks = chunk_page_text(raw_text, page_num)
        all_chunks.extend(t_chunks)

        has_caption = bool(
            re.search(
                r"\b(Table\s+\d+|Figure\s+\d+|Box\s+\d+|Annex\s+\d+|Source:|Priority Matrix|Key damages|Key emission|Unit\s+#|Urgency)\b",
                raw_text,
            )
        )
        # Skip front matter TOC/lists (pages 1-7) from visual element extraction
        if page_num >= 8 and (has_caption or n_imgs > 0):
            visual_candidate_pages.append(page_num)

    print(
        f"Text chunks: {len(all_chunks)} | Visual candidate pages: {len(visual_candidate_pages)} | Skipped: {len(skipped_pages)}",
        flush=True,
    )

    vision_cache_path = PROJECT_ROOT / ".cache" / "adv_vision_cache.json"
    vision_cache_path.parent.mkdir(parents=True, exist_ok=True)
    vision_cache: dict[str, Any] = (
        json.loads(vision_cache_path.read_text(encoding="utf-8"))
        if vision_cache_path.exists()
        else {}
    )

    visual_elements: list[dict[str, Any]] = []
    vision_in_tok = 0
    vision_out_tok = 0
    kind_counts = {"table": 0, "chart": 0, "image_diagram": 0}

    for pnum in visual_candidate_pages:
        png_path = PAGES_DIR / f"p{pnum:03d}.png"
        elems, in_t, out_t = extract_visual_elements_for_page(
            client,
            settings.agent_model,
            pnum,
            png_path,
            page_texts[str(pnum)],
            vision_cache,
            vision_cache_path,
        )
        vision_in_tok += in_t
        vision_out_tok += out_t
        for el_i, el in enumerate(elems):
            kind = el["kind"]
            kind_counts[kind] += 1
            crop_rel = crop_element(
                png_path,
                el.get("bbox_pct"),
                CROPS_DIR / f"p{pnum:03d}_el{el_i:02d}.png",
            )
            title = str(el.get("title", "")).strip()
            summary = str(el.get("summary", "")).strip()
            cid = f"ADV:{DOC_ID}:p{pnum:03d}:v{el_i:02d}"
            rec = {
                "id": cid,
                "source_id": DOC_ID,
                "source": DOC_SOURCE,
                "page": pnum,
                "element_type": kind,
                "text": f"[{kind.upper()} | Page {pnum}] {title}\n{summary}".strip(),
                "metadata": {
                    "page": pnum,
                    "element_type": kind,
                    "title": title,
                    "bbox_pct": el.get("bbox_pct"),
                    "crop_image": crop_rel,
                    "page_image": f"starter/data/advanced/pages/p{pnum:03d}.png",
                },
            }
            visual_elements.append(rec)
            all_chunks.append(rec)
        print(f"  Vision page {pnum}: extracted {len(elems)} element(s)", flush=True)

    (ADV_DIR / "page_texts.json").write_text(
        json.dumps(page_texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ADV_DIR / "visual_elements.json").write_text(
        json.dumps(visual_elements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ADV_DIR / "adv_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    embed_chars = 0
    if not args.skip_embed:
        emb_cache_path = PROJECT_ROOT / ".cache" / "adv_embeddings_cache.json"
        vectors, embed_chars = embed_all(
            client,
            settings.embedding_model,
            [c["text"] for c in all_chunks],
            emb_cache_path,
        )
        store_pgvector(settings, all_chunks, vectors)
        print(f"Upserted {len(all_chunks)} multimodal chunks into pgvector table adv_chunks", flush=True)

    elapsed = time.perf_counter() - t0
    stats = {
        "document_id": DOC_ID,
        "title": "Vietnam Country Climate and Development Report (World Bank, July 2022)",
        "source_url": DOC_URL,
        "license": "World Bank Open Knowledge / Public Disclosure Authorized (CC BY 3.0 IGO)",
        "total_physical_pages": total_pages,
        "indexed_pages": total_pages - len(skipped_pages),
        "skipped_pages": skipped_pages,
        "text_chunks": sum(1 for c in all_chunks if c["element_type"] == "text"),
        "visual_elements": len(visual_elements),
        "visual_elements_by_kind": kind_counts,
        "total_indexed_records": len(all_chunks),
        "vision_input_tokens": vision_in_tok,
        "vision_output_tokens": vision_out_tok,
        "vision_cost_usd": round(estimate_llm_cost_usd(vision_in_tok, vision_out_tok), 6),
        "embed_chars": embed_chars,
        "embed_cost_usd": round(estimate_embed_cost_usd(embed_chars), 6),
        "elapsed_seconds": round(elapsed, 2),
    }
    (ADV_DIR / "ingest_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
