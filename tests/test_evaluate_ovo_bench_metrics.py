import unittest

from scripts.evaluate_ovo_bench import exclude_prefixed_videos, summarize


class EvaluateOvoBenchMetricTests(unittest.TestCase):
    def test_warmup_prefix_is_excluded_from_official_summary_input(self):
        rows = [
            {"video_id": "warmup-a"},
            {"video_id": "ovo-1"},
        ]

        kept, excluded = exclude_prefixed_videos(rows, ["warmup-"])

        self.assertEqual([row["video_id"] for row in kept], ["ovo-1"])
        self.assertEqual(excluded, 1)

    def test_wall_clock_scopes_are_reported_without_changing_legacy_total(self):
        row = {
            "video_id": "video-1",
            "benchmark_task": "ACR",
            "ovo_official_score": 1,
            "ovo_strict_score": 1,
            "load_video_sec": "2.0",
            "init_prompt_sec": "1.0",
            "qa_sec": "4.0",
            "elapsed_video_sec": "17.0",
            "cumulative_encode_video_sec": "10.0",
            "loaded_frames": "20",
            "sample_fps": "2.0",
            "semantic_input_frames": "20",
            "vit_output_policy": "structured_residual",
            "vit_output_budget_per_frame": "121",
            "vit_output_base_tokens_per_frame": "100",
            "vit_output_residual_tokens_per_frame": "21",
            "semantic_proposal_sec": "0.5",
            "semantic_preprocess_sec": "1.5",
            "semantic_embedding_sec": "1.0",
            "semantic_verification_sec": "0.5",
            "semantic_vit_encoder_sec": "2.0",
            "semantic_context_write_sec": "3.0",
        }

        result = summarize([row])

        self.assertEqual(result["total_encode_video_sec"], 10.0)
        self.assertEqual(result["wall_clock_sec"]["video_encode"], 10.0)
        self.assertEqual(
            result["wall_clock_sec"]["online_video_processing"],
            10.0,
        )
        self.assertEqual(
            result["wall_clock_sec"]["online_model_pipeline"],
            15.0,
        )
        self.assertEqual(result["wall_clock_sec"]["video_load"], 2.0)
        self.assertEqual(result["wall_clock_sec"]["init_prompt"], 1.0)
        self.assertEqual(result["wall_clock_sec"]["qa"], 4.0)
        self.assertEqual(result["wall_clock_sec"]["full_pipeline"], 17.0)
        self.assertEqual(result["latency_scope_sec"]["model_encoding"], 3.0)
        self.assertEqual(result["latency_scope_sec"]["visual_encoding"], 4.5)
        self.assertEqual(result["latency_scope_sec"]["stream_ingestion"], 8.5)
        self.assertEqual(
            result["realtime_metrics"]["observed_stream_duration_sec"],
            10.0,
        )
        self.assertEqual(result["realtime_metrics"]["online_processing_fps"], 2.0)
        self.assertEqual(result["realtime_metrics"]["realtime_compute_ratio"], 1.0)
        self.assertEqual(result["arrived_frames"], 20)
        self.assertEqual(
            result["vit_output_reduction"]["policy"],
            "structured_residual",
        )
        self.assertEqual(
            result["vit_output_reduction"]["residual_tokens_per_frame"],
            21,
        )
        self.assertEqual(
            result["wall_clock_sec"]["video_encode_unprofiled"],
            1.5,
        )
        self.assertIn("full_pipeline", result["metric_definitions"])
        self.assertEqual(
            result["paper_reporting_policy"]["primary_latency_metric"],
            "wall_clock_sec.online_video_processing",
        )
        self.assertIn(
            "已解码",
            result["paper_reporting_policy"]["system_input_contract"],
        )
        self.assertIn(
            "wall_clock_sec.full_pipeline",
            result["paper_reporting_policy"]["excluded_from_speedup"],
        )
        self.assertIn(
            "video_decode",
            result["paper_reporting_policy"]["excluded_from_speedup"],
        )


if __name__ == "__main__":
    unittest.main()
