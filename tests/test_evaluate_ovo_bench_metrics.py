import unittest

from scripts.evaluate_ovo_bench import summarize


class EvaluateOvoBenchMetricTests(unittest.TestCase):
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
        self.assertEqual(result["wall_clock_sec"]["video_load"], 2.0)
        self.assertEqual(result["wall_clock_sec"]["init_prompt"], 1.0)
        self.assertEqual(result["wall_clock_sec"]["qa"], 4.0)
        self.assertEqual(result["wall_clock_sec"]["full_pipeline"], 17.0)
        self.assertEqual(result["latency_scope_sec"]["model_encoding"], 3.0)
        self.assertEqual(result["latency_scope_sec"]["visual_encoding"], 4.5)
        self.assertEqual(result["latency_scope_sec"]["stream_ingestion"], 8.5)
        self.assertEqual(
            result["wall_clock_sec"]["video_encode_unprofiled"],
            1.5,
        )
        self.assertIn("full_pipeline", result["metric_definitions"])


if __name__ == "__main__":
    unittest.main()
