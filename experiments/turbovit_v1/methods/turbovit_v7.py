from dataclasses import dataclass
from math import isqrt
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
class TurboV7FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0
    semantic_stability: float = 1.0
    adaptive_ratio: float = 1.0
    probe_ms: float = 0.0
    token_selector_ms: float = 0.0
    rolling_reuse_ratio: float = 1.0
    long_reuse_ratio: float = 0.0
    segment_count: int = 0
    mean_segment_len: float = 0.0
    segment_expansion_ratio: float = 1.0


def _infer_patch_grid(seq_len: int) -> Tuple[int, int]:
    grid = isqrt(seq_len)
    if grid * grid == seq_len:
        return grid, 0
    patch_tokens = seq_len - 1
    grid = isqrt(patch_tokens)
    if grid * grid != patch_tokens:
        raise ValueError(
            f"expected square patch grid with or without CLS token, got seq_len={seq_len}"
        )
    return grid, 1


def _segments_from_mask(mask: torch.Tensor, max_gap: int, min_segment_len: int) -> Tuple[List[Tuple[int, int, int]], int]:
    grid = mask.shape[0]
    raw_count = int(mask.sum().item())
    segments = []

    for row in range(grid):
        cols = torch.nonzero(mask[row], as_tuple=False).flatten().tolist()
        if not cols:
            continue
        start = cols[0]
        prev = cols[0]
        for col in cols[1:]:
            if col - prev <= max_gap + 1:
                prev = col
            else:
                segments.append((row, start, prev + 1))
                start = col
                prev = col
        segments.append((row, start, prev + 1))

    expanded = []
    for row, start, end in segments:
        length = end - start
        if min_segment_len > 1 and length < min_segment_len:
            pad = min_segment_len - length
            left = pad // 2
            right = pad - left
            start = max(0, start - left)
            end = min(grid, end + right)
            if end - start < min_segment_len:
                start = max(0, end - min_segment_len)
                end = min(grid, start + min_segment_len)
        expanded.append((row, start, end))

    return expanded, raw_count


def _select_dynamic_segments(
    token_stability: torch.Tensor,
    dynamic_ratio: float,
    segment_max_gap: int,
    min_segment_len: int,
) -> Tuple[torch.Tensor, int, float, float]:
    if token_stability.shape[0] != 1:
        raise ValueError("v7 prototype currently expects batch size 1")
    if segment_max_gap < 0:
        raise ValueError("segment_max_gap must be >= 0")
    if min_segment_len < 1:
        raise ValueError("min_segment_len must be >= 1")

    seq_len = token_stability.shape[1]
    grid, prefix_tokens = _infer_patch_grid(seq_len)
    patch_count = seq_len - prefix_tokens
    target_patches = max(1, min(patch_count, int(round(patch_count * dynamic_ratio))))
    patch_scores = token_stability[:, prefix_tokens:]
    selected = torch.topk(patch_scores, k=target_patches, dim=1, largest=False).indices[0]
    mask_flat = torch.zeros(patch_count, dtype=torch.bool, device=token_stability.device)
    mask_flat[selected] = True
    mask = mask_flat.view(grid, grid)

    segments, raw_count = _segments_from_mask(mask, segment_max_gap, min_segment_len)
    dynamic_tokens = set(range(prefix_tokens))
    segment_lens = []
    for row, start, end in segments:
        segment_lens.append(end - start)
        for col in range(start, end):
            dynamic_tokens.add(prefix_tokens + row * grid + col)

    dynamic_indices = torch.tensor(sorted(dynamic_tokens), device=token_stability.device).view(1, -1)
    segment_count = len(segments)
    mean_segment_len = float(sum(segment_lens) / segment_count) if segment_count else 0.0
    expansion_ratio = float((dynamic_indices.numel() - prefix_tokens) / raw_count) if raw_count else 1.0
    return dynamic_indices, segment_count, mean_segment_len, expansion_ratio


@torch.inference_mode()
def encode_stream_turbovit_v7(
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
    segment_max_gap: int = 1,
    min_segment_len: int = 2,
    warmup_frames: int = 2,
) -> List[TurboV7FrameResult]:
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
        segment_count = 0
        mean_segment_len = 0.0
        segment_expansion_ratio = 1.0

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
                segment_expansion_ratio = 0.0
            else:
                adaptive_ratio = _adaptive_ratio(
                    semantic_stability,
                    sparse_ratio_min,
                    sparse_ratio_max,
                    skip_feature_threshold,
                    dense_feature_threshold,
                )
                dyn_indices, segment_count, mean_segment_len, segment_expansion_ratio = _select_dynamic_segments(
                    token_stability,
                    adaptive_ratio,
                    segment_max_gap,
                    min_segment_len,
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
            TurboV7FrameResult(
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
                segment_count=segment_count,
                mean_segment_len=mean_segment_len,
                segment_expansion_ratio=segment_expansion_ratio,
            )
        )
    return results
