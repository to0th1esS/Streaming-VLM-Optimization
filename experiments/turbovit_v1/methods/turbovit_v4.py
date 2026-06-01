from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult


@dataclass
class TurboV4FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0
    semantic_stability: float = 1.0
    semantic_to_rolling: float = 1.0
    semantic_to_long: float = 1.0
    adaptive_ratio: float = 1.0
    probe_ms: float = 0.0


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left - right) ** 2).item())


def _forward_probe(model, frame: torch.Tensor, probe_layer: int) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    embedding = model.embed(frame)
    hidden_states = embedding
    caches = []
    last_idx = min(probe_layer, len(model.blocks) - 1)
    for layer_idx in range(last_idx + 1):
        hidden_states, cache = model.blocks[layer_idx].forward_with_cache(hidden_states)
        caches.append(cache)
    return embedding, hidden_states, caches


def _finish_from_prefix(
    model,
    hidden_states: torch.Tensor,
    prefix_caches: List[Dict[str, torch.Tensor]],
    start_layer: int,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    caches = list(prefix_caches)
    for layer_idx in range(start_layer, len(model.blocks)):
        hidden_states, cache = model.blocks[layer_idx].forward_with_cache(hidden_states)
        caches.append(cache)
    return model.norm(hidden_states), caches


def _sparse_from_prefix(
    model,
    hidden_states: torch.Tensor,
    prefix_caches: List[Dict[str, torch.Tensor]],
    ref_caches: List[Dict[str, torch.Tensor]],
    start_layer: int,
    dynamic_ratio: float,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], float, float, float]:
    next_caches = list(prefix_caches)
    selector_ms = 0.0
    sparse_compute_ms = 0.0
    dynamic_counts = []
    token_counts = []

    for layer_idx in range(start_layer, len(model.blocks)):
        block = model.blocks[layer_idx]
        ref_cache = ref_caches[layer_idx]

        selector_start = perf_counter()
        residual = hidden_states
        normed = block.norm1(hidden_states)
        q, key, value = block._project_qkv(normed)
        similarity = F.cosine_similarity(key, ref_cache["key"], dim=-1)
        seq_len = hidden_states.shape[1]
        num_dynamic = max(1, min(seq_len, int(round(seq_len * dynamic_ratio))))
        dynamic_indices = torch.topk(similarity, k=num_dynamic, dim=1, largest=False).indices
        gather_idx = dynamic_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        selector_ms += (perf_counter() - selector_start) * 1000.0

        sparse_start = perf_counter()
        q_selected = q.gather(1, gather_idx)
        residual_selected = residual.gather(1, gather_idx)
        q_heads = block._split_heads(q_selected)
        k_heads = block._split_heads(key)
        v_heads = block._split_heads(value)
        scale = q_heads.shape[-1] ** -0.5
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_selected = torch.matmul(attn_probs, v_heads)
        attn_selected = block._merge_heads(attn_selected)
        attn_selected = block.attn.out_proj(attn_selected)
        hidden_selected = residual_selected + attn_selected
        mlp_selected = block.mlp(block.norm2(hidden_selected))
        output_selected = hidden_selected + mlp_selected
        hidden_states = ref_cache["output"].clone()
        hidden_states.scatter_(1, gather_idx, output_selected)
        sparse_compute_ms += (perf_counter() - sparse_start) * 1000.0

        next_caches.append(
            {
                "key": key.detach(),
                "output": hidden_states.detach(),
            }
        )
        dynamic_counts.append(num_dynamic)
        token_counts.append(seq_len)

    output = model.norm(hidden_states)
    observed_ratio = float(sum(dynamic_counts) / sum(token_counts)) if token_counts else 1.0
    return output, next_caches, selector_ms, sparse_compute_ms, observed_ratio


def _adaptive_ratio(
    stability: float,
    sparse_ratio_min: float,
    sparse_ratio_max: float,
    skip_feature_threshold: float,
    dense_feature_threshold: float,
) -> float:
    if skip_feature_threshold <= dense_feature_threshold:
        return sparse_ratio_max
    alpha = (skip_feature_threshold - stability) / (skip_feature_threshold - dense_feature_threshold)
    alpha = max(0.0, min(1.0, alpha))
    return sparse_ratio_min + alpha * (sparse_ratio_max - sparse_ratio_min)


@torch.inference_mode()
def encode_stream_turbovit_v4(
    model,
    video: torch.Tensor,
    refresh_interval: int = 4,
    sparse_ratio_min: float = 0.75,
    sparse_ratio_max: float = 1.0,
    probe_layer: int = 2,
    skip_patch_threshold: float = 0.001,
    dense_patch_threshold: float = 0.006,
    skip_feature_threshold: float = 0.9995,
    dense_feature_threshold: float = 0.98,
    warmup_frames: int = 2,
) -> List[TurboV4FrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if not (0.0 < sparse_ratio_min <= sparse_ratio_max <= 1.0):
        raise ValueError("sparse ratios must satisfy 0 < min <= max <= 1")
    if skip_patch_threshold > dense_patch_threshold:
        raise ValueError("skip_patch_threshold must be <= dense_patch_threshold")
    if dense_feature_threshold > skip_feature_threshold:
        raise ValueError("dense_feature_threshold must be <= skip_feature_threshold")

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
    rolling_embed = None
    rolling_output = None
    rolling_probe = None
    long_embed = None
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
        semantic_to_rolling = 1.0
        semantic_to_long = 1.0
        semantic_stability = 1.0
        adaptive_ratio = 1.0

        if forced_reference:
            output, rolling_caches = model.forward_with_caches(frame)
            current_embed = model.embed(frame)
            rolling_embed = current_embed.detach()
            rolling_output = output.detach()
            rolling_probe = rolling_caches[min(probe_layer, len(rolling_caches) - 1)]["output"].detach()
            long_embed = rolling_embed
            long_probe = rolling_probe
            frame_drift = 0.0
            decision = "dense"
            is_reference = True
        else:
            probe_start = perf_counter()
            current_embed, probe_output, prefix_caches = _forward_probe(model, frame, probe_layer)
            probe_ms = (perf_counter() - probe_start) * 1000.0
            frame_drift = _mse(current_embed, rolling_embed) if rolling_embed is not None else 0.0
            semantic_to_rolling = _cosine(probe_output, rolling_probe) if rolling_probe is not None else 1.0
            semantic_to_long = _cosine(probe_output, long_probe) if long_probe is not None else 1.0
            semantic_stability = min(semantic_to_rolling, semantic_to_long)
            start_layer = min(probe_layer, len(model.blocks) - 1) + 1

            if frame_drift >= dense_patch_threshold or semantic_stability < dense_feature_threshold:
                output, rolling_caches = _finish_from_prefix(model, probe_output, prefix_caches, start_layer)
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                rolling_probe = probe_output.detach()
                long_embed = rolling_embed
                long_probe = rolling_probe
                decision = "dense"
                is_reference = True
            elif frame_drift <= skip_patch_threshold and semantic_stability >= skip_feature_threshold:
                output = rolling_output
                decision = "skip"
                is_reference = False
                observed_ratio = 0.0
                adaptive_ratio = 0.0
            else:
                adaptive_ratio = _adaptive_ratio(
                    semantic_stability,
                    sparse_ratio_min,
                    sparse_ratio_max,
                    skip_feature_threshold,
                    dense_feature_threshold,
                )
                output, rolling_caches, selector_ms, sparse_compute_ms, observed_ratio = _sparse_from_prefix(
                    model,
                    probe_output,
                    prefix_caches,
                    rolling_caches,
                    start_layer,
                    adaptive_ratio,
                )
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                rolling_probe = prefix_caches[-1]["output"].detach()
                decision = "sparse"
                is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV4FrameResult(
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
                semantic_to_rolling=semantic_to_rolling,
                semantic_to_long=semantic_to_long,
                adaptive_ratio=adaptive_ratio,
                probe_ms=probe_ms,
            )
        )
    return results
