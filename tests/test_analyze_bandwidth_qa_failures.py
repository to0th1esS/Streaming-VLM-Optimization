import unittest

from scripts.analyze_bandwidth_qa_failures import build_report


def make_row(video_id, task, baseline_score, loaded_frames=100):
    return {
        "video_id": video_id,
        "benchmark_group": "backward",
        "benchmark_task": task,
        "official_id": video_id,
        "query_index": "0",
        "ovo_official_score": str(baseline_score),
        "loaded_frames": str(loaded_frames),
        "semantic_kept_frames": "10",
        "semantic_recency_kept_frames": "4",
        "semantic_written_tokens": "1210",
        "cumulative_encode_video_sec": "1.5",
    }


class AnalyzeBandwidthQaFailuresTests(unittest.TestCase):
    def test_reports_flips_and_recency_share(self):
        baseline = [
            make_row("a", "HLD", 1, 200),
            make_row("b", "ASI", 0, 100),
        ]
        candidate = [
            make_row("a", "HLD", 0, 200),
            make_row("b", "ASI", 1, 100),
        ]

        rows, report = build_report(baseline, candidate)

        self.assertEqual(rows[0]["flip"], "negative_flip")
        self.assertEqual(rows[1]["flip"], "positive_flip")
        self.assertAlmostEqual(rows[0]["recency_share"], 0.4)
        self.assertEqual(report["by_task"]["HLD"]["negative_flips"], 1)
        self.assertEqual(report["by_task"]["ASI"]["positive_flips"], 1)


if __name__ == "__main__":
    unittest.main()
