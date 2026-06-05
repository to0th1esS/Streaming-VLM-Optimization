import unittest

import torch

from model.vit_patch import _raw_rgb_candidate_indices, _raw_rgb_signatures
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

    def test_grid_sample_signature_reads_fixed_spatial_grid(self):
        video = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
        video[0, 1, 1] = torch.tensor([255, 0, 0], dtype=torch.uint8)
        video[0, 1, 3] = torch.tensor([0, 255, 0], dtype=torch.uint8)
        video[0, 3, 1] = torch.tensor([0, 0, 255], dtype=torch.uint8)
        video[0, 3, 3] = torch.tensor([255, 255, 255], dtype=torch.uint8)

        signatures = _raw_rgb_signatures(
            video,
            grid_size=2,
            mode="grid_sample",
        )

        self.assertEqual(tuple(signatures.shape), (2, 12))
        self.assertAlmostEqual(float(signatures[0].norm()), 1.0, places=6)
        self.assertEqual(float(signatures[1].norm()), 0.0)

    def test_unknown_raw_signature_mode_fails(self):
        with self.assertRaises(ValueError):
            _raw_rgb_signatures(
                torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
                mode="unknown",
            )

    def test_saliency_gate_falls_back_to_periodic_slot_without_peak(self):
        gate = SemanticStreamGate(
            refresh_interval=1000,
            selection_policy="budget_topk",
            budget_window_size=4,
            budget_keep_per_window=1,
        )
        signatures = torch.tensor([[1.0, 0.0]] * 8)

        selected = _raw_rgb_candidate_indices(
            signatures,
            gate,
            candidate_multiplier=1,
            proposal_policy="saliency_gated",
            saliency_z_threshold=1.5,
        )

        self.assertEqual(selected.tolist(), [0, 4])

    def test_saliency_gate_replaces_periodic_slot_for_clear_peak(self):
        gate = SemanticStreamGate(
            refresh_interval=1000,
            selection_policy="budget_topk",
            budget_window_size=4,
            budget_keep_per_window=1,
        )
        signatures = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
            ]
        )

        selected = _raw_rgb_candidate_indices(
            signatures,
            gate,
            candidate_multiplier=1,
            proposal_policy="saliency_gated",
            saliency_z_threshold=1.5,
        )

        self.assertEqual(selected.tolist(), [0, 6])

    def test_unknown_raw_proposal_policy_fails(self):
        gate = SemanticStreamGate(
            refresh_interval=1000,
            selection_policy="budget_topk",
            budget_window_size=4,
        )
        with self.assertRaises(ValueError):
            _raw_rgb_candidate_indices(
                torch.tensor([[1.0, 0.0]]),
                gate,
                candidate_multiplier=1,
                proposal_policy="unknown",
            )


if __name__ == "__main__":
    unittest.main()
