from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import torch
import torch.nn.functional as F

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult


@dataclass
class TurboV2FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0


def _sparse_from_reference(model, frame: torch.Tensor, ref_caches, dynamic_ratio: float):
    hidden_states = model.embed(frame)
    next_caches = []
    selector_ms = 0.0
    sparse_compute_ms = 0.0
    dynamic_counts = []
    token_counts = []

    for block, ref_cache in zip(model.blocks, ref_caches):
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
    dynamic_ratio_observed = float(sum(dynamic_counts) / sum(token_counts))
    return output, next_caches, selector_ms, sparse_compute_ms, dynamic_ratio_observed


@torch.inference_mode()
def encode_stream_turbovit_v2(
    model,
    video: torch.Tensor,
    refresh_interval: int = 4,
    dynamic_ratio: float = 0.75,
    skip_threshold: float = 0.0005,
    dense_threshold: float = 0.006,
    warmup_frames: int = 2,
) -> List[TurboV2FrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if not (0.0 < dynamic_ratio <= 1.0):
        raise ValueError("dynamic_ratio must be in (0, 1]")
    if skip_threshold < 0 or dense_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    if skip_threshold > dense_threshold:
        raise ValueError("skip_threshold must be <= dense_threshold")

    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        model.forward_with_caches(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    results = []
    ref_caches: List[Dict[str, torch.Tensor]] = []
    ref_embed = None
    ref_output = None

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        forced_reference = (frame_idx % refresh_interval == 0) or not ref_caches

        _synchronize_if_needed(device)
        start = perf_counter()
        selector_ms = 0.0
        sparse_compute_ms = 0.0
        dynamic_ratio_observed = 1.0
        frame_drift = 0.0

        current_embed = model.embed(frame)
        if ref_embed is not None:
            frame_drift = float(torch.mean((current_embed - ref_embed) ** 2).item())

        if forced_reference or frame_drift >= dense_threshold:
            output, ref_caches = model.forward_with_caches(frame)
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            decision = "dense"
            is_reference = True
        elif frame_drift <= skip_threshold and ref_output is not None:
            output = ref_output
            decision = "skip"
            is_reference = False
            dynamic_ratio_observed = 0.0
        else:
            # Pay per-layer selector cost only when a cheap frame-level test says
            # the frame is neither static nor too different.
            output, ref_caches, selector_ms, sparse_compute_ms, dynamic_ratio_observed = _sparse_from_reference(
                model,
                frame,
                ref_caches,
                dynamic_ratio,
            )
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            decision = "sparse"
            is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV2FrameResult(
                frame_idx=frame_idx,
                is_reference=is_reference,
                latency_ms=latency_ms,
                selector_ms=selector_ms,
                sparse_compute_ms=sparse_compute_ms,
                dynamic_ratio_observed=dynamic_ratio_observed,
                output=output.detach().cpu(),
                decision=decision,
                frame_drift=frame_drift,
            )
        )
    return results
