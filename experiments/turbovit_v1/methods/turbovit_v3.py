from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v1 import TurboFrameResult
from experiments.turbovit_v1.methods.turbovit_v2 import _sparse_from_reference


@dataclass
class TurboV3FrameResult(TurboFrameResult):
    decision: str = "dense"
    frame_drift: float = 0.0
    patch_mse_to_long: float = 0.0
    rolling_long_mse: float = 0.0
    feature_gate_ms: float = 0.0
    feature_gate_cos_to_rolling: float = 1.0
    feature_gate_cos_to_long: float = 1.0
    feature_gate_cos_min: float = 1.0


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left - right) ** 2).item())


def _forward_to_gate(
    model,
    frame: torch.Tensor,
    gate_layer: int,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    hidden_states = model.embed(frame)
    caches = []
    last_idx = min(gate_layer, len(model.blocks) - 1)
    for layer_idx in range(last_idx + 1):
        hidden_states, cache = model.blocks[layer_idx].forward_with_cache(hidden_states)
        caches.append(cache)
    return hidden_states, caches


def _finish_from_gate(
    model,
    hidden_states: torch.Tensor,
    prefix_caches: List[Dict[str, torch.Tensor]],
    gate_layer: int,
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    caches = list(prefix_caches)
    start_layer = min(gate_layer, len(model.blocks) - 1) + 1
    for layer_idx in range(start_layer, len(model.blocks)):
        hidden_states, cache = model.blocks[layer_idx].forward_with_cache(hidden_states)
        caches.append(cache)
    return model.norm(hidden_states), caches


def _cache_gate_output(caches: List[Dict[str, torch.Tensor]], gate_layer: int) -> Optional[torch.Tensor]:
    if not caches:
        return None
    return caches[min(gate_layer, len(caches) - 1)]["output"].detach()


@torch.inference_mode()
def encode_stream_turbovit_v3(
    model,
    video: torch.Tensor,
    refresh_interval: int = 4,
    dynamic_ratio: float = 0.9,
    skip_threshold: float = 0.001,
    dense_threshold: float = 0.006,
    feature_gate_layer: int = 5,
    feature_skip_threshold: float = 0.98,
    warmup_frames: int = 2,
) -> List[TurboV3FrameResult]:
    if refresh_interval < 1:
        raise ValueError("refresh_interval must be >= 1")
    if not (0.0 < dynamic_ratio <= 1.0):
        raise ValueError("dynamic_ratio must be in (0, 1]")
    if skip_threshold < 0 or dense_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    if skip_threshold > dense_threshold:
        raise ValueError("skip_threshold must be <= dense_threshold")
    if not (0.0 <= feature_skip_threshold <= 1.0):
        raise ValueError("feature_skip_threshold must be in [0, 1]")

    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        model.forward_with_caches(video[frame_idx : frame_idx + 1])
        _forward_to_gate(model, video[frame_idx : frame_idx + 1], feature_gate_layer)
    _synchronize_if_needed(device)

    results = []
    ref_caches: List[Dict[str, torch.Tensor]] = []
    ref_embed = None
    ref_output = None
    rolling_gate_output = None
    long_embed = None
    long_gate_output = None

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        forced_reference = (frame_idx % refresh_interval == 0) or not ref_caches

        _synchronize_if_needed(device)
        start = perf_counter()
        selector_ms = 0.0
        sparse_compute_ms = 0.0
        feature_gate_ms = 0.0
        dynamic_ratio_observed = 1.0
        feature_gate_cos_to_rolling = 1.0
        feature_gate_cos_to_long = 1.0
        feature_gate_cos_min = 1.0

        current_embed = model.embed(frame)
        frame_drift = _mse(current_embed, ref_embed) if ref_embed is not None else 0.0
        patch_mse_to_long = _mse(current_embed, long_embed) if long_embed is not None else 0.0
        rolling_long_mse = _mse(ref_embed, long_embed) if ref_embed is not None and long_embed is not None else 0.0

        if forced_reference or frame_drift >= dense_threshold:
            output, ref_caches = model.forward_with_caches(frame)
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            rolling_gate_output = _cache_gate_output(ref_caches, feature_gate_layer)
            long_embed = current_embed.detach()
            long_gate_output = rolling_gate_output
            decision = "dense"
            is_reference = True
        elif frame_drift <= skip_threshold and ref_output is not None:
            gate_start = perf_counter()
            gate_output, gate_caches = _forward_to_gate(model, frame, feature_gate_layer)
            if rolling_gate_output is not None:
                feature_gate_cos_to_rolling = _cosine(gate_output, rolling_gate_output)
            if long_gate_output is not None:
                feature_gate_cos_to_long = _cosine(gate_output, long_gate_output)
            feature_gate_cos_min = min(feature_gate_cos_to_rolling, feature_gate_cos_to_long)
            feature_gate_ms = (perf_counter() - gate_start) * 1000.0

            if feature_gate_cos_min >= feature_skip_threshold:
                output = ref_output
                decision = "skip"
                is_reference = False
                dynamic_ratio_observed = 0.0
            else:
                output, ref_caches = _finish_from_gate(
                    model,
                    gate_output,
                    gate_caches,
                    feature_gate_layer,
                )
                ref_embed = current_embed.detach()
                ref_output = output.detach()
                rolling_gate_output = _cache_gate_output(ref_caches, feature_gate_layer)
                long_embed = current_embed.detach()
                long_gate_output = rolling_gate_output
                decision = "gate_dense"
                is_reference = True
        else:
            output, ref_caches, selector_ms, sparse_compute_ms, dynamic_ratio_observed = _sparse_from_reference(
                model,
                frame,
                ref_caches,
                dynamic_ratio,
            )
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            rolling_gate_output = _cache_gate_output(ref_caches, feature_gate_layer)
            decision = "sparse"
            is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            TurboV3FrameResult(
                frame_idx=frame_idx,
                is_reference=is_reference,
                latency_ms=latency_ms,
                selector_ms=selector_ms,
                sparse_compute_ms=sparse_compute_ms,
                dynamic_ratio_observed=dynamic_ratio_observed,
                output=output.detach().cpu(),
                decision=decision,
                frame_drift=frame_drift,
                patch_mse_to_long=patch_mse_to_long,
                rolling_long_mse=rolling_long_mse,
                feature_gate_ms=feature_gate_ms,
                feature_gate_cos_to_rolling=feature_gate_cos_to_rolling,
                feature_gate_cos_to_long=feature_gate_cos_to_long,
                feature_gate_cos_min=feature_gate_cos_min,
            )
        )
    return results
