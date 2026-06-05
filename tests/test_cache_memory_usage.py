import unittest
from types import SimpleNamespace

import torch

from model.abstract_rekv import Abstract_ReKV


class CacheMemoryUsageTest(unittest.TestCase):
    def test_counts_cache_tensors_without_double_counting_gpu_views(self):
        model = object.__new__(Abstract_ReKV)
        cpu_block = SimpleNamespace(
            cpu_data=(torch.zeros(2, 3), torch.zeros(2, 3)),
            gpu_data=torch.zeros(2, 2, 3),
        )
        layer = SimpleNamespace(
            length=17,
            local_k=torch.zeros(2, 3),
            local_v=torch.zeros(2, 3),
            init_k=torch.zeros(1),
            init_v=torch.zeros(1),
            global_remainder=(torch.zeros(2), torch.zeros(2)),
            global_buffer=torch.zeros(4),
            cuda_cache=SimpleNamespace(data=torch.zeros(8)),
            block_k=[SimpleNamespace(data=torch.zeros(5))],
            global_blocks=[[cpu_block]],
        )
        model.kv_cache = [layer]

        usage = model.calc_cache_memory_usage()

        expected_gpu_elements = 6 + 6 + 1 + 1 + 2 + 2 + 4 + 8 + 5
        expected_cpu_elements = 6 + 6
        self.assertEqual(usage["gpu_bytes"], expected_gpu_elements * 4)
        self.assertEqual(usage["cpu_bytes"], expected_cpu_elements * 4)
        self.assertEqual(
            usage["total_bytes"],
            (expected_gpu_elements + expected_cpu_elements) * 4,
        )
        self.assertEqual(usage["logical_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
