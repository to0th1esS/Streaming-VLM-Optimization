from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticVideoConfig:
    num_frames: int = 32
    image_size: int = 64
    channels: int = 3
    drift_per_frame: float = 0.015
    noise_std: float = 0.01
    seed: int = 0


def make_redundant_video(config: SyntheticVideoConfig) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.seed)
    base = torch.rand(
        config.channels,
        config.image_size,
        config.image_size,
        generator=generator,
    )
    velocity = torch.randn(base.shape, generator=generator) * config.drift_per_frame

    frames = []
    for frame_idx in range(config.num_frames):
        noise = torch.randn(base.shape, generator=generator) * config.noise_std
        frame = (base + frame_idx * velocity + noise).clamp(0.0, 1.0)
        frames.append(frame)
    return torch.stack(frames, dim=0)
