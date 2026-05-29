from typing import Dict, List

import torch
import torch.nn.functional as F


def adjacent_layer_cosine(dense_results) -> List[Dict[str, float]]:
    if len(dense_results) < 2:
        return []

    depth = len(dense_results[0].layer_outputs)
    rows = []
    for layer_idx in range(depth):
        sims = []
        for frame_idx in range(1, len(dense_results)):
            prev_layer = dense_results[frame_idx - 1].layer_outputs[layer_idx].flatten(0, 1)
            cur_layer = dense_results[frame_idx].layer_outputs[layer_idx].flatten(0, 1)
            sim = F.cosine_similarity(prev_layer, cur_layer, dim=-1).mean()
            sims.append(float(sim.item()))
        rows.append(
            {
                "layer": layer_idx,
                "adjacent_cosine_mean": float(torch.tensor(sims).mean().item()),
                "adjacent_cosine_std": float(torch.tensor(sims).std(unbiased=False).item()),
            }
        )
    return rows
