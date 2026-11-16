"""Unit tests for Day 7 retrievers, filters, adjacent chunks, and agent helpers."""
import unittest

from src.agent import _parse_final_json
from src.retrieval_eval import is_gold_hit, pack_context
from src.retrievers import AdjacentChunkStore, Bm25Retriever, TfidfRetriever, infer_year_filter


SAMPLE_CHUNKS = [
    {
        "id": "A:doc23:p011:c000",
        "source_id": "doc23",
        "source": "Taking Stock 2023",
        "page": 11,
        "metadata": {"year": 2023, "section": "overview"},
        "text": "Vietnam real GDP growth slowed to 3.7 percent year-on-year in H1 2023.",
    },
    {
        "id": "A:doc23:p011:c001",
        "source_id": "doc23",
        "source": "Taking Stock 2023",
        "page": 11,
        "metadata": {"year": 2023, "section": "overview"},
        "text": "Public investment disbursement reached 30.5 percent of the annual budget.",
    },
    {
        "id": "A:doc25:p009:c000",
        "source_id": "doc25",
        "source": "Taking Stock 2025",
        "page": 9,
        "metadata": {"year": 2025, "section": "overview"},
        "text": "Vietnam GDP growth accelerated to 7.1 percent in 2024 and is projected at 6.8 percent in 2025.",
    },
]


class Day7PipelineTests(unittest.TestCase):
    def test_infer_year_filter_only_on_explicit_edition(self) -> None:
        self.assertEqual(
            infer_year_filter("In the 2023 report, what was projected GDP growth for 2025?"),
            {"year": 2023},
        )
        self.assertEqual(
            infer_year_filter("According to the March 2025 report, what was inflation?"),
            {"year": 2025},
        )
        self.assertIsNone(infer_year_filter("What was Vietnam GDP growth in 2023?"))

    def test_bm25_and_tfidf_respect_filter(self) -> None:
        bm25 = Bm25Retriever(SAMPLE_CHUNKS)
        tfidf = TfidfRetriever(SAMPLE_CHUNKS)
        hits_bm25 = bm25.search("GDP growth", filters={"year": 2025}, top_k=3)
        hits_tfidf = tfidf.search("GDP growth", filters={"year": 2025}, top_k=3)
        self.assertEqual(len(hits_bm25), 1)
        self.assertEqual(hits_bm25[0]["id"], "A:doc25:p009:c000")
        self.assertEqual(len(hits_tfidf), 1)
        self.assertEqual(hits_tfidf[0]["id"], "A:doc25:p009:c000")

    def test_adjacent_chunk_store(self) -> None:
        store = AdjacentChunkStore(SAMPLE_CHUNKS)
        adj = store.get_adjacent("A:doc23:p011:c000", window=1)
        self.assertEqual([c["id"] for c in adj], ["A:doc23:p011:c001"])

    def test_norm_gold_hit_and_pack_context(self) -> None:
        chunk_with_newlines = {
            **SAMPLE_CHUNKS[0],
            "text": "Vietnam real GDP\ngrowth slowed to\n3.7 percent year-on-year in H1 2023.",
        }
        gold = [{"source_id": "doc23", "page": 11, "evidence": "growth slowed to 3.7 percent"}]
        self.assertTrue(is_gold_hit(chunk_with_newlines, gold))
        ctx, ids = pack_context([SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[1]], budget=400)
        self.assertEqual(ids, ["A:doc23:p011:c000", "A:doc23:p011:c001"])
        self.assertIn("[A:doc23:p011:c000 | Taking Stock 2023 | page 11]", ctx)

    def test_parse_final_json_handles_fences(self) -> None:
        raw = '```json\n{"status": "answered", "answer": "3.7% [A:doc23:p011:c000]", "citations": ["A:doc23:p011:c000"]}\n```'
        parsed = _parse_final_json(raw)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual(parsed["citations"], ["A:doc23:p011:c000"])


if __name__ == "__main__":
    unittest.main()
