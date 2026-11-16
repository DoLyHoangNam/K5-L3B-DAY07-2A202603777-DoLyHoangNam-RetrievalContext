"""Unit tests for Day 7 Advanced Challenge multimodal benchmark and helpers."""
import json
import unittest
from collections import Counter
from pathlib import Path

from src.adv_ingest import chunk_page_text

ADV_CASES_PATH = Path(__file__).resolve().parents[1] / "starter" / "data" / "advanced" / "adv_cases.json"


class AdvancedPipelineTests(unittest.TestCase):
    def test_chunk_page_text_preserves_page_and_ids(self) -> None:
        text = "Vietnam climate resilience investment needs " * 40
        chunks = chunk_page_text(text, page=18, size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len({c["id"] for c in chunks}), len(chunks))
        self.assertTrue(all(c["page"] == 18 and c["element_type"] == "text" for c in chunks))

    def test_adv_benchmark_meets_advanced_requirements(self) -> None:
        cases = json.loads(ADV_CASES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len({c["id"] for c in cases}), len(cases))

        dev = [c for c in cases if c["split"] == "dev"]
        test = [c for c in cases if c["split"] == "test"]
        self.assertGreaterEqual(sum(1 for c in dev if c["answerable"]), 15)
        self.assertGreaterEqual(sum(1 for c in dev if not c["answerable"]), 3)
        self.assertGreaterEqual(sum(1 for c in test if c["answerable"]), 24)
        self.assertGreaterEqual(sum(1 for c in test if not c["answerable"]), 6)

        counts = Counter(c["group"] for c in test)
        self.assertGreaterEqual(counts["table"], 6)
        self.assertGreaterEqual(counts["image_diagram"], 5)
        self.assertGreaterEqual(counts["chart"], 5)
        self.assertGreaterEqual(counts["multi_hop"], 4)
        self.assertGreaterEqual(counts["text"], 4)
        self.assertGreaterEqual(counts["unanswerable"], 6)


if __name__ == "__main__":
    unittest.main()
