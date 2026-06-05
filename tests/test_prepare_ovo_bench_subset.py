import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_ovo_bench_subset import load_excluded_sources


class PrepareOvoBenchSubsetTest(unittest.TestCase):
    def test_exclusion_tracks_ids_and_original_videos(self):
        rows = [
            {
                "official_id": 995,
                "original_video": "YouTube_Games/shared.mp4",
            },
            {
                "official_id": 1000,
                "original_video": "",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "subset.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )

            source_ids, original_videos = load_excluded_sources(path)

        self.assertEqual(source_ids, {995, 1000})
        self.assertEqual(original_videos, {"YouTube_Games/shared.mp4"})

    def test_empty_exclusion_returns_empty_sets(self):
        source_ids, original_videos = load_excluded_sources("")

        self.assertEqual(source_ids, set())
        self.assertEqual(original_videos, set())


if __name__ == "__main__":
    unittest.main()
