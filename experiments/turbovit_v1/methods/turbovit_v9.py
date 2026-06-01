from dataclasses import dataclass
from time import perf_counter
from typing import List

import torch

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult
from experiments.turbovit_v1.methods.turbovit_v4 import _mse


@dataclass
class TurboV9FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0
    embed_ms: float = 0.0


@torch.inference_mode()
def encode_stream_turbovit_v9(
    model,
    video: torch.Tensor,
    refresh_interval: int = 8,
    skip_patch_threshold: float = 0.01,
    warmup_frames: int = 2,
) -> List[TurboV9FrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if skip_patch_threshold < 0:
        raise ValueError("skip_patch_threshold must be >= 0")

    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        model.forward_with_caches(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    results = []
    rolling_embed = None
    rolling_output = None

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        forced_reference = (frame_idx % refresh_interval == 0) or rolling_output is None

        _synchronize_if_needed(device)
        start = perf_counter()
        embed_ms = 0.0

        if forced_reference:
            output, _ = model.forward_with_caches(frame)
            rolling_embed = model.embed(frame).detach()
            rolling_output = output.detach()
            decision = "dense"
            is_reference = True
            frame_drift = 0.0
        else:
            embed_start = perf_counter()
            current_embed = model.embed(frame)
            _synchronize_if_needed(device)
            embed_ms = (perf_counter() - embed_start) * 1000.0
            frame_drift = _mse(current_embed, rolling_embed)
            if frame_drift <= skip_patch_threshold:
                output = rolling_output
                decision = "skip"
                is_reference = False
            else:
                output, _ = model.forward_with_caches(frame)
                rolling_embed = current_embed.detach()
                rolling_output = output.detach()
                decision = "dense"
                is_reference = True

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV9FrameResult(
                frame_idx=frame_idx,
                is_reference=is_reference,
                latency_ms=latency_ms,
                selector_ms=0.0,
                sparse_compute_ms=0.0,
                dynamic_ratio_observed=0.0 if decision == "skip" else 1.0,
                output=output.detach().cpu(),
                decision=decision,
                frame_drift=frame_drift,
                embed_ms=embed_ms,
            )
        )

    return results
