import unittest

import torch

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

    def test_reference_frame_consumes_first_window_budget(self):
        gate = SemanticStreamGate(
            refresh_interval=1000,
            selection_policy="budget_topk",
            budget_window_size=4,
            budget_keep_per_window=1,
        )
        signatures = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.2, 0.8],
                    [-1.0, 0.0],
                ]
            ),
            dim=-1,
        )

        selected = gate.select_indices_from_window_signatures(
            signatures,
            token_count=10,
        )

        self.assertEqual(selected.tolist(), [0, 7])
        self.assertEqual(gate.stats["kept_frames"], 2)
        self.assertEqual(gate.stats["budget_kept_frames"], 1)


if __name__ == "__main__":
    unittest.main()
