import unittest

from model.vision_accelerator.context import InferenceContext


class InferenceContextTest(unittest.TestCase):
    def test_step_preserves_stream_position_across_calls(self):
        context = InferenceContext(cache_interval=3)

        modes = []
        for _ in range(5):
            context.step()
            modes.append(context.is_reference_chunk)

        self.assertEqual(modes, [True, False, False, True, False])
        self.assertEqual(context.processed_units, 5)
        self.assertEqual(context.dense_units, 2)
        self.assertEqual(context.sparse_units, 3)

    def test_reset_restores_first_reference(self):
        context = InferenceContext(cache_interval=2)
        context.step()
        context.step()

        context.reset()
        context.step()

        self.assertTrue(context.is_reference_chunk)
        self.assertEqual(context.chunk_idx, 0)
        self.assertEqual(context.processed_units, 1)
        self.assertEqual(context.dense_units, 1)
        self.assertEqual(context.sparse_units, 0)


if __name__ == "__main__":
    unittest.main()
