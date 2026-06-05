import unittest

from model.vision_accelerator.semantic_stream import SemanticStreamGate


class SemanticStreamPeriodicTest(unittest.TestCase):
    def test_periodic_selection_happens_by_frame_index(self):
        gate = SemanticStreamGate(
            refresh_interval=4,
            recency_keep_frames=2,
            selection_policy="periodic",
        )
        gate.set_recency_window(0, 10)

        selected = gate.select_periodic_indices(total_frames=10, token_count=100)

        self.assertEqual(selected.tolist(), [0, 4, 8, 9])
        self.assertEqual(gate.stats["input_frames"], 10)
        self.assertEqual(gate.stats["kept_frames"], 4)
        self.assertEqual(gate.stats["written_tokens"], 400)
        self.assertEqual(gate.stats["recency_kept_frames"], 1)

    def test_periodic_selection_preserves_global_index_across_batches(self):
        gate = SemanticStreamGate(
            refresh_interval=4,
            recency_keep_frames=2,
            selection_policy="periodic",
        )
        gate.select_periodic_indices(total_frames=10, token_count=1)
        gate.set_recency_window(10, 15)

        selected = gate.select_periodic_indices(total_frames=5, token_count=1)

        self.assertEqual(selected.tolist(), [2, 3, 4])
        self.assertEqual(gate.frame_idx, 15)


if __name__ == "__main__":
    unittest.main()
