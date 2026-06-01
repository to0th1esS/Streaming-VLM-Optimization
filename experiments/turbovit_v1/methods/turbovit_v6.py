from dataclasses import dataclass
from math import ceil, isqrt
from time import perf_counter
from typing import Dict, List, Tuple

import torch

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult
from experiments.turbovit_v1.methods.turbovit_v4 import _finish_from_prefix, _forward_probe, _mse
from experiments.turbovit_v1.methods.turbovit_v5 import (
    _adaptive_ratio,
    _anchor_token_scores,
    _dual_anchor_sparse_from_prefix,
)


@dataclass
class TurboV6FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0
    semantic_stability: float = 1.0
    adaptive_ratio: float = 1.0
    probe_ms: float = 0.0
    token_selector_ms: float = 0.0
    rolling_reuse_ratio: float = 1.0
    long_reuse_ratio: float = 0.0
    group_size: int = 1
    dynamic_group_ratio: float = 1.0


def _infer_patch_grid(seq_len: int) -> int:
    patch_tokens = seq_len - 1
    grid = isqrt(patch_tokens)
    if grid * grid != patch_tokens:
        raise ValueError(f"expected square patch grid after CLS token, got {patch_tokens} patch tokens")
    return grid


def _select_dynamic_token_groups(
    token_stability: torch.Tensor,
    dynamic_ratio: float,
    group_size: int,
    group_score: str,
) -> Tuple[torch.Tensor, float]:
    if token_stability.shape[0] != 1:
        raise ValueError("v6 prototype currently expects batch size 1")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if group_score not in {"mean", "min"}:
        raise ValueError("group_score must be one of: mean, min")

    seq_len = token_stability.shape[1]
    grid = _infer_patch_grid(seq_len)
    patch_scores = token_stability[0, 1:].view(grid, grid)

    groups = []
    for row in range(0, grid, group_size):
        for col in range(0, grid, group_size):
            block = patch_scores[row : row + group_size, col : col + group_size]
            score = block.mean() if group_score == "mean" else block.min()
            token_indices = []
            for rr in range(row, min(row + group_size, grid)):
                for cc in range(col, min(col + group_size, grid)):
                    token_indices.append(1 + rr * grid + cc)
            groups.append((score, token_indices))

    target_patches = max(1, min(seq_len - 1, int(round((seq_len - 1) * dynamic_ratio))))
    mean_group_tokens = max(1, group_size * group_size)
    num_groups = max(1, min(len(groups), ceil(target_patches / mean_group_tokens)))
    group_scores = torch.stack([item[0] for item in groups]).view(1, -1)
    selected_groups = torch.topk(group_scores, k=num_groups, dim=1, largest=False).indices[0].tolist()

    dynamic_tokens = {0}
    for group_idx in selected_groups:
        dynamic_tokens.update(groups[group_idx][1])
    dynamic_indices = torch.tensor(sorted(dynamic_tokens), device=token_stability.device).view(1, -1)
    dynamic_group_ratio = float(num_groups / len(groups)) if groups else 1.0
    return dynamic_indices, dynamic_group_ratio


@torch.inference_mode()
def encode_stream_turbovit_v6(
    model,
    video: torch.Tensor,
    refresh_interval: int = 8,
    sparse_ratio_min: float = 0.6,
    sparse_ratio_max: float = 0.95,
    probe_layer: int = 2,
    skip_patch_threshold: float = 0.001,
    dense_patch_threshold: float = 0.006,
    skip_feature_threshold: float = 0.9999,
    dense_feature_threshold: float = 0.98,
    anchor_mode: str = "dual",
    group_size: int = 2,
    group_score: str = "mean",
    warmup_frames: int = 2,
) -> List[TurboV6FrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if not (0.0 < sparse_ratio_min <= sparse_ratio_max <= 1.0):
        raise ValueError("sparse ratios must satisfy 0 < min <= max <= 1")
    if skip_patch_threshold > dense_patch_threshold:
        raise ValueError("skip_patch_threshold must be <= dense_patch_threshold")
    if dense_feature_threshold > skip_feature_threshold:
        raise ValueError("dense_feature_threshold must be <= skip_feature_threshold")
    if anchor_mode not in {"dual", "rolling_only", "long_only"}:
        raise ValueError("anchor_mode must be one of: dual, rolling_only, long_only")

    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        frame = video[frame_idx : frame_idx + 1]
        model.forward_with_caches(frame)
        _forward_probe(model, frame, probe_layer)
    _synchronize_if_needed(device)

    results = []
    rolling_caches: List[Dict[str, torch.Tensor]] = []
    long_caches: List[Dict[str, torch.Tensor]] = []
    rolling_embed = None
    rolling_output = None
    rolling_probe = None
    long_probe = None

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        forced_reference = (frame_idx % refresh_interval == 0) or not rolling_caches

        _synchronize_if_needed(device)
        start = perf_counter()
        selector_ms = 0.0
        sparse_compute_ms = 0.0
        observed_ratio = 1.0
        probe_ms = 0.0
        token_selector_ms = 0.0
        semantic_stability = 1.0
        adaptive_ratio = 1.0
        rolling_reuse_ratio = 1.0
        long_reuse_ratio = 0.0
        dynamic_group_ratio = 1.0

        if forced_reference:
            output, rolling_caches = model.forward_with_caches(frame)
            current_embed = model.embed(frame)
            rolling_embed = current_embed.detach()
            rolling_output = output.detach()
            rolling_probe = rolling_caches[min(probe_layer, len(rolling_caches) - 1)]["output"].detach()
            long_caches = rolling_caches
            long_probe = rolling_probe
            frame_drift = 0.0
            decision = "dense"
            is_reference = True
        else:
            probe_start = perf_counter()
            current_embed, probe_output, prefix_caches = _forward_probe(model, frame, probe_layer)
            probe_ms = (perf_counter() - probe_start) * 1000.0
            frame_drift = _mse(current_embed, rolling_embed) if rolling_embed is not None else 0.0
            start_layer = min(probe_layer, len(model.blocks) - 1) + 1

            selector_start = perf_counter()
            token_stability, use_rolling, rolling_reuse_ratio, long_reuse_ratio = _anchor_token_scores(
                probe_output,
                rolling_probe,
                long_probe,
                anchor_mode,
            )
            semantic_stability = float(token_stability.mean().item())
            token_selector_ms = (perf_counter() - selector_start) * 1000.0

            if frame_drift >= dense_patch_threshold or semantic_stability < dense_feature_threshold:
                output, rolling_caches = _finish_from_prefix(model, probe_output, prefix_caches, start_layer)
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                rolling_probe = probe_output.detach()
                long_caches = rolling_caches
                long_probe = rolling_probe
                decision = "dense"
                is_reference = True
            elif frame_drift <= skip_patch_threshold and semantic_stability >= skip_feature_threshold:
                output = rolling_output
                decision = "skip"
                is_reference = False
                observed_ratio = 0.0
                adaptive_ratio = 0.0
                dynamic_group_ratio = 0.0
            else:
                adaptive_ratio = _adaptive_ratio(
                    semantic_stability,
                    sparse_ratio_min,
                    sparse_ratio_max,
                    skip_feature_threshold,
                    dense_feature_threshold,
                )
                dyn_indices, dynamic_group_ratio = _select_dynamic_token_groups(
                    token_stability,
                    adaptive_ratio,
                    group_size,
                    group_score,
                )
                output, rolling_caches, selector_ms, sparse_compute_ms, observed_ratio = _dual_anchor_sparse_from_prefix(
                    model,
                    probe_output,
                    prefix_caches,
                    rolling_caches,
                    long_caches,
                    start_layer,
                    dyn_indices,
                    use_rolling,
                )
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                rolling_probe = prefix_caches[-1]["output"].detach()
                decision = "sparse"
                is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV6FrameResult(
                frame_idx=frame_idx,
                is_reference=is_reference,
                latency_ms=latency_ms,
                selector_ms=selector_ms,
                sparse_compute_ms=sparse_compute_ms,
                dynamic_ratio_observed=observed_ratio,
                output=output.detach().cpu(),
                decision=decision,
                frame_drift=frame_drift,
                semantic_stability=semantic_stability,
                adaptive_ratio=adaptive_ratio,
                probe_ms=probe_ms,
                token_selector_ms=token_selector_ms,
                rolling_reuse_ratio=rolling_reuse_ratio,
                long_reuse_ratio=long_reuse_ratio,
                group_size=group_size,
                dynamic_group_ratio=dynamic_group_ratio,
            )
        )
    return results
