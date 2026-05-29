from dataclasses import dataclass
from time import perf_counter
from typing import List

import torch


@dataclass
class DenseFrameResult:
    frame_idx: int
    latency_ms: float
    output: torch.Tensor
    layer_outputs: List[torch.Tensor]


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def encode_stream_dense(model, video: torch.Tensor, warmup_frames: int = 2) -> List[DenseFrameResult]:
    model.eval()
    device = next(model.parameters()).device
    video = video.to(device)

    for frame_idx in range(min(warmup_frames, video.shape[0])):
        model.forward_with_layers(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    results = []
    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        _synchronize_if_needed(device)
        start = perf_counter()
        output, layer_outputs = model.forward_with_layers(frame)
        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        results.append(
            DenseFrameResult(
                frame_idx=frame_idx,
                latency_ms=latency_ms,
                output=output.detach().cpu(),
                layer_outputs=[layer.detach().cpu() for layer in layer_outputs],
            )
        )
    return results
