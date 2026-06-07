import unittest

import torch

from model.vision_accelerator.token_reducer import (
    FixedBudgetTokenReducer,
    StructuredGridTokenReducer,
)


class FixedBudgetTokenReducerTest(unittest.TestCase):
    def test_uniform_policy_keeps_fixed_budget(self):
        reducer = FixedBudgetTokenReducer(
            output_token_budget=4,
            coverage_tokens=2,
            policy="uniform",
        )
        features = torch.arange(16, dtype=torch.float32).view(2, 8, 1)

        reduced = reducer(features, batch_size=1, frames=2)

        self.assertEqual(tuple(reduced.shape), (2, 4, 1))
        self.assertEqual(reducer.stats["input_tokens"], 16)
        self.assertEqual(reducer.stats["output_tokens"], 8)

    def test_coverage_innovation_keeps_changed_token(self):
        reducer = FixedBudgetTokenReducer(
            output_token_budget=4,
            coverage_tokens=2,
            policy="coverage_innovation",
        )
        first = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        second = first.clone()
        second[3] = torch.tensor([0.0, 1.0])

        reducer(first.unsqueeze(0), batch_size=1, frames=1)
        reduced = reducer(second.unsqueeze(0), batch_size=1, frames=1)

        self.assertTrue(
            torch.any(torch.all(reduced[0] == second[3], dim=-1)).item()
        )
        self.assertEqual(reducer.stats["innovation_tokens"], 2)

    def test_reset_clears_rolling_anchor_and_stats(self):
        reducer = FixedBudgetTokenReducer(
            output_token_budget=2,
            coverage_tokens=1,
            policy="coverage_innovation",
        )
        reducer(torch.ones(1, 4, 2), batch_size=1, frames=1)

        reducer.reset()

        self.assertIsNone(reducer.previous_features)
        self.assertEqual(reducer.stats["frames"], 0)
        self.assertEqual(reducer.stats["output_tokens"], 0)

    def test_selection_features_can_differ_from_output_features(self):
        reducer = FixedBudgetTokenReducer(
            output_token_budget=2,
            coverage_tokens=1,
            policy="coverage_innovation",
        )
        output = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        reference = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
        )
        changed = reference.clone()
        changed[0, 2] = torch.tensor([-1.0, 0.0])

        reducer(output, frames=1, selection_features=reference)
        selected = reducer(output, frames=1, selection_features=changed)

        self.assertTrue(torch.equal(selected[0, 1], output[0, 2]))

    def test_drift_feature_sketch_uses_requested_dimension(self):
        reducer = FixedBudgetTokenReducer(
            output_token_budget=2,
            coverage_tokens=1,
            policy="coverage_innovation",
            drift_feature_dims=2,
        )
        features = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)

        sketch = reducer._drift_features(features)

        self.assertEqual(tuple(sketch.shape), (1, 4, 2))
        self.assertTrue(torch.equal(sketch, features[..., [0, 3]]))


class StructuredGridTokenReducerTest(unittest.TestCase):
    def test_structured_pool_keeps_regular_output_grid(self):
        reducer = StructuredGridTokenReducer(output_token_budget=4)
        features = torch.arange(
            2 * 9 * 3,
            dtype=torch.float32,
        ).reshape(2, 9, 3)

        reduced = reducer(features, batch_size=1, frames=2)

        self.assertEqual(tuple(reduced.shape), (2, 4, 3))
        self.assertTrue(reduced.is_contiguous())
        self.assertEqual(reducer.stats["input_tokens"], 392)
        self.assertEqual(reducer.stats["output_tokens"], 8)

    def test_structured_pool_rejects_non_square_budget(self):
        with self.assertRaisesRegex(ValueError, "perfect-square"):
            StructuredGridTokenReducer(output_token_budget=8)


if __name__ == "__main__":
    unittest.main()
