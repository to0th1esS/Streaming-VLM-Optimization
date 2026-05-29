from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import torch
import torch.nn.functional as F

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed


@dataclass
class TurboFrameResult:
    frame_idx: int
    is_reference: bool
    latency_ms: float
    selector_ms: float
    sparse_compute_ms: float
    dynamic_ratio_observed: float
    output: torch.Tensor


@torch.inference_mode()
def encode_stream_turbovit_v1(
    model,
    video: torch.Tensor,
    refresh_interval: int = 4,
    dynamic_ratio: float = 0.5,
    warmup_frames: int = 2,
) -> List[TurboFrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if not (0.0 < dynamic_ratio <= 1.0):
        raise ValueError("dynamic_ratio must be in (0, 1]")

    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        model.forward_with_caches(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    results = []
    ref_caches: List[Dict[str, torch.Tensor]] = []
    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        is_reference = (frame_idx % refresh_interval == 0) or not ref_caches

        _synchronize_if_needed(device)
        start = perf_counter()
        selector_ms = 0.0
        sparse_compute_ms = 0.0
        dynamic_counts = []
        token_counts = []

        if is_reference:
            output, ref_caches = model.forward_with_caches(frame)
        else:
            hidden_states = model.embed(frame)
            next_caches = []
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
            ref_caches = next_caches

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        dynamic_ratio_observed = (
            float(sum(dynamic_counts) / sum(token_counts)) if token_counts else 1.0
        )
        results.append(
            TurboFrameResult(
                frame_idx=frame_idx,
                is_reference=is_reference,
                latency_ms=latency_ms,
                selector_ms=selector_ms,
                sparse_compute_ms=sparse_compute_ms,
                dynamic_ratio_observed=dynamic_ratio_observed,
                output=output.detach().cpu(),
            )
        )
    return results
