import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_ovo_bench import evaluate_rows, summarize
from scripts.prepare_ovo_bench_subset import (
    convert_annotations,
    evenly_spaced_indices,
    select_query_indices,
    select_source_items,
)
from scripts.summarize_ovo_bench_validation import summarize as summarize_validation


class OVOBenchAdapterTest(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            {
                "id": 1,
                "task": "EPM",
                "video": "source/a.mp4",
                "realtime": 12,
                "question": "What happened?",
                "options": ["first", "second"],
                "gt": 1,
            },
            {
                "id": 2,
                "task": "REC",
                "video": "source/b.mp4",
                "activity": "jumping",
                "test_info": [
                    {"realtime": 5, "count": 1},
                    {"realtime": 8, "count": 2},
                ],
            },
            {
                "id": 3,
                "task": "SSR",
                "video": "source/c.mp4",
                "test_info": [
                    {"realtime": 7, "step": "open the box", "type": 1},
                ],
            },
            {
                "id": 4,
                "task": "CRR",
                "video": "source/d.mp4",
                "question": "What is in the box?",
                "test_info": [
                    {"realtime": 9, "type": 0},
                ],
            },
        ]

    def test_conversion_uses_official_chunk_names_and_answers(self):
        rows, counts = convert_annotations(
            self.annotations,
            chunked_dir="/data/chunks",
            tasks=["EPM", "REC", "SSR", "CRR"],
            max_queries_per_source=1,
        )

        self.assertEqual(counts["EPM"], 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["video_path"], "/data/chunks/1.mp4")
        self.assertEqual(rows[0]["conversations"][0]["answer"], "B")
        self.assertEqual(rows[1]["video_path"], "/data/chunks/2_0.mp4")
        self.assertEqual(rows[1]["conversations"][0]["answer"], "1")
        self.assertEqual(rows[2]["conversations"][0]["answer"], "Yes")
        self.assertEqual(rows[3]["conversations"][0]["answer"], "No")

    def test_source_item_and_query_limits_are_independent(self):
        rows, _ = convert_annotations(
            self.annotations,
            chunked_dir="/data/chunks",
            tasks=["REC"],
            max_source_items_per_task=1,
            max_queries_per_source=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["query_index"] for row in rows], [0, 1])

    def test_duration_stratified_source_selection_covers_range(self):
        items = [
            {
                "id": index,
                "task": "EPM",
                "realtime": duration,
            }
            for index, duration in enumerate((80, 10, 60, 30, 50, 20), start=1)
        ]

        selected = select_source_items(items, 3, "duration_stratified")

        self.assertEqual(
            [item["realtime"] for item in selected],
            [10, 30, 80],
        )

    def test_time_stratified_query_selection_keeps_early_and_late_queries(self):
        queries = [
            {"realtime": value}
            for value in (50, 10, 40, 20, 30)
        ]

        selected = select_query_indices(queries, 3, "time_stratified")

        self.assertEqual(selected, [1, 4, 0])
        self.assertEqual(evenly_spaced_indices(5, 3), [0, 2, 4])

    def test_stratified_conversion_preserves_original_query_indices(self):
        annotation = {
            "id": 5,
            "task": "REC",
            "video": "source/e.mp4",
            "activity": "turning",
            "test_info": [
                {"realtime": value, "count": index}
                for index, value in enumerate((50, 10, 40, 20, 30))
            ],
        }

        rows, _ = convert_annotations(
            [annotation],
            chunked_dir="/data/chunks",
            tasks=["REC"],
            max_source_items_per_task=1,
            max_queries_per_source=3,
            source_selection="duration_stratified",
            query_selection="time_stratified",
        )

        self.assertEqual([row["query_index"] for row in rows], [1, 4, 0])
        self.assertEqual(
            [row["video_path"] for row in rows],
            [
                "/data/chunks/5_1.mp4",
                "/data/chunks/5_4.mp4",
                "/data/chunks/5_0.mp4",
            ],
        )

    def test_official_and_strict_scores_are_reported(self):
        rows = [
            {
                "video_id": "a",
                "benchmark_task": "EPM",
                "answer": "B",
                "pred_answer": "The answer is B.",
            },
            {
                "video_id": "b",
                "benchmark_task": "REC",
                "answer": "2",
                "pred_answer": "2",
            },
            {
                "video_id": "c",
                "benchmark_task": "SSR",
                "answer": "Yes",
                "pred_answer": "Yes",
            },
        ]
        evaluated = evaluate_rows(rows)
        summary = summarize(evaluated)

        self.assertTrue(all(row["ovo_official_score"] == 1 for row in evaluated))
        self.assertTrue(all(row["ovo_strict_score"] == 1 for row in evaluated))
        self.assertEqual(summary["official_three_group_average"], 1.0)

    def test_validation_summary_uses_dense_encode_time(self):
        template = {
            "samples": 3,
            "official_three_group_average": 1.0,
            "strict_three_group_average": 1.0,
            "semantic_input_frames": 10,
            "semantic_kept_frames": 5,
            "semantic_token_reduction": 0.5,
            "per_group": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for method, encode_sec in (
                ("dense", 10.0),
                ("periodic", 2.0),
                ("hybrid_cm2", 1.0),
            ):
                method_dir = root / method
                method_dir.mkdir()
                metrics = {**template, "total_encode_video_sec": encode_sec}
                (method_dir / "metrics.json").write_text(
                    json.dumps(metrics),
                    encoding="utf-8",
                )

            rows = summarize_validation(
                root,
                ["dense", "periodic", "hybrid_cm2"],
            )

        self.assertEqual(rows[1]["speedup_vs_dense"], 5.0)
        self.assertEqual(rows[2]["speedup_vs_dense"], 10.0)


if __name__ == "__main__":
    unittest.main()
