from typing import Dict, List

import torch
import torch.nn.functional as F


def compare_outputs(dense_results, turbo_results) -> List[Dict[str, float]]:
    rows = []
    for dense, turbo in zip(dense_results, turbo_results):
        dense_flat = dense.output.flatten()
        turbo_flat = turbo.output.flatten()
        cosine = F.cosine_similarity(dense_flat, turbo_flat, dim=0)
        mse = torch.mean((dense_flat - turbo_flat) ** 2)
        rows.append(
            {
                "frame_idx": dense.frame_idx,
                "is_reference": int(turbo.is_reference),
                "output_cosine": float(cosine.item()),
                "output_mse": float(mse.item()),
            }
        )
    return rows
