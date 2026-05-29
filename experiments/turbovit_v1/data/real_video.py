from pathlib import Path
from urllib.request import urlretrieve

import imageio.v3 as iio
import torch
import torch.nn.functional as F


DEFAULT_VIDEO_URL = "https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4"


def ensure_video(video_path: Path, url: str = DEFAULT_VIDEO_URL) -> Path:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        urlretrieve(url, video_path)
    return video_path


def load_video_frames(
    video_path: Path,
    num_frames: int = 24,
    image_size: int = 64,
    stride: int = 2,
) -> torch.Tensor:
    frames = []
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if frame_idx % stride != 0:
            continue
        tensor = torch.from_numpy(frame).float() / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        tensor = tensor[..., :3].permute(2, 0, 1)
        frames.append(tensor)
        if len(frames) >= num_frames:
            break

    if len(frames) < num_frames:
        raise RuntimeError(f"Only decoded {len(frames)} frames from {video_path}; need {num_frames}.")

    video = torch.stack(frames, dim=0)
    video = F.interpolate(
        video,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return video.clamp(0.0, 1.0)
