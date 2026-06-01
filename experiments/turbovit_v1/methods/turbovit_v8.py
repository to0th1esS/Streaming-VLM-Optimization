from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import torch

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult
from experiments.turbovit_v1.methods.turbovit_v4 import _finish_from_prefix, _forward_probe, _mse
from experiments.turbovit_v1.methods.turbovit_v5 import _adaptive_ratio, _anchor_token_scores
from experiments.turbovit_v1.methods.turbovit_v7 import _select_dynamic_segments


@dataclass
class TurboV8FrameResult(TurboFrameResult):
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
    kv_projection_ms: float = 0.0


def _layer_kv_reuse_sparse_from_prefix(
    model,
    hidden_states: torch.Tensor,
    prefix_caches: List[Dict[str, torch.Tensor]],
    rolling_caches: List[Dict[str, torch.Tensor]],
    long_caches: List[Dict[str, torch.Tensor]],
    start_layer: int,
    dynamic_indices: torch.Tensor,
    use_rolling_tokens: torch.Tensor,
    anchor_mix_mode: str,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], float, float, float, float]:
    next_caches = list(prefix_caches)
    selector_ms = 0.0
    sparse_compute_ms = 0.0
    kv_projection_ms = 0.0
    dynamic_counts = []
    token_counts = []

    gather_idx = dynamic_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    use_rolling_expanded = use_rolling_tokens.unsqueeze(-1)

    for layer_idx in range(start_layer, len(model.blocks)):
        block = model.blocks[layer_idx]
        rolling_cache = rolling_caches[layer_idx]
        long_cache = long_caches[layer_idx]

        sparse_start = perf_counter()
        residual_selected = hidden_states.gather(1, gather_idx)

        kv_start = perf_counter()
        normed_selected = block.norm1(residual_selected)
        q_selected, key_selected, value_selected = block._project_qkv(normed_selected)
        kv_projection_ms += (perf_counter() - kv_start) * 1000.0

        base_key = _mix_anchor_cache("key", rolling_cache, long_cache, use_rolling_tokens, anchor_mix_mode)
        base_value = _mix_anchor_cache("value", rolling_cache, long_cache, use_rolling_tokens, anchor_mix_mode)
        key = base_key.scatter(1, gather_idx, key_selected)
        value = base_value.scatter(1, gather_idx, value_selected)

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

        base_output = _mix_anchor_cache("output", rolling_cache, long_cache, use_rolling_tokens, anchor_mix_mode)
        hidden_states = base_output.scatter(1, gather_idx, output_selected)
        sparse_compute_ms += (perf_counter() - sparse_start) * 1000.0

        next_caches.append(
            {
                "key": key.detach(),
                "value": value.detach(),
                "output": hidden_states.detach(),
            }
        )
        dynamic_counts.append(dynamic_indices.shape[1])
        token_counts.append(hidden_states.shape[1])

    output = model.norm(hidden_states)
    observed_ratio = float(sum(dynamic_counts) / sum(token_counts)) if token_counts else 1.0
    return output, next_caches, selector_ms, sparse_compute_ms, observed_ratio, kv_projection_ms


def _mix_anchor_cache(
    cache_name: str,
    rolling_cache: Dict[str, torch.Tensor],
    long_cache: Dict[str, torch.Tensor],
    use_rolling_tokens: torch.Tensor,
    anchor_mix_mode: str,
) -> torch.Tensor:
    if anchor_mix_mode == "where" or use_rolling_tokens.shape[0] != 1:
        return torch.where(
            use_rolling_tokens.unsqueeze(-1),
            rolling_cache[cache_name],
            long_cache[cache_name],
        ).clone()
    if anchor_mix_mode != "scatter":
        raise ValueError("anchor_mix_mode must be one of: where, scatter")

    rolling_ratio = float(use_rolling_tokens.float().mean().item())
    if rolling_ratio >= 0.5:
        base = rolling_cache[cache_name].clone()
        other_cache = long_cache[cache_name]
        other_mask = ~use_rolling_tokens[0]
    else:
        base = long_cache[cache_name].clone()
        other_cache = rolling_cache[cache_name]
        other_mask = use_rolling_tokens[0]

    other_indices = torch.nonzero(other_mask, as_tuple=False).flatten()
    if other_indices.numel() > 0:
        idx = other_indices.view(1, -1, 1).expand(-1, -1, base.shape[-1])
        base.scatter_(1, idx, other_cache.gather(1, idx))
    return base


@torch.inference_mode()
def encode_stream_turbovit_v8(
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
    anchor_mix_mode: str = "where",
    segment_max_gap: int = 1,
    min_segment_len: int = 2,
    warmup_frames: int = 2,
) -> List[TurboV8FrameResult]:
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
    if anchor_mix_mode not in {"where", "scatter"}:
        raise ValueError("anchor_mix_mode must be one of: where, scatter")

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
        kv_projection_ms = 0.0

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
                (
                    output,
                    rolling_caches,
                    selector_ms,
                    sparse_compute_ms,
                    observed_ratio,
                    kv_projection_ms,
                ) = _layer_kv_reuse_sparse_from_prefix(
                    model,
                    probe_output,
                    prefix_caches,
                    rolling_caches,
                    long_caches,
                    start_layer,
                    dyn_indices,
                    use_rolling,
                    anchor_mix_mode,
                )
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                rolling_probe = prefix_caches[-1]["output"].detach()
                decision = "sparse"
                is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV8FrameResult(
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
                kv_projection_ms=kv_projection_ms,
            )
        )
    return results
