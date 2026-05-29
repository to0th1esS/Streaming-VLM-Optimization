import json
import pathlib
import sys
import types
import argparse
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model.vit_patch import vit_patch_hf


class DummySelfAttention(torch.nn.Module):
    def __init__(self, embed_dim=8, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = 0.0
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)


class DummyLayer(torch.nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = 0.0
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.self_attn = DummySelfAttention(embed_dim=embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        return (hidden_states,)


class DummyVisionTower(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.vision_model = types.SimpleNamespace(
            encoder=types.SimpleNamespace(layers=torch.nn.ModuleList([layer]))
        )

    def forward(self, pixel_values, output_hidden_states=False):
        batch_size = pixel_values.shape[0]
        hidden_states = torch.randn(batch_size, 5, 8, device=pixel_values.device)
        for layer in self.vision_model.encoder.layers:
            hidden_states = layer(hidden_states, None, False)[0]
        return types.SimpleNamespace(hidden_states=[hidden_states])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = DummyVisionTower(DummyLayer())
        self.config = types.SimpleNamespace(
            vision_feature_layer=0,
            vision_feature_select_strategy="default",
        )
        self.multi_modal_projector = torch.nn.Linear(8, 8)
        self.encoded_chunks = []

    def apply_pooling(self, video_features):
        return video_features

    def _get_video_features(self, pixel_values_videos):
        raise RuntimeError("vit_patch_hf should replace this method")

    def _encode_video_chunk(self, video_chunk):
        frames = video_chunk.shape[0]
        pixel_values = torch.randn(1, frames, 3, 2, 2)
        features = self._get_video_features(pixel_values)
        self.encoded_chunks.append(features.shape)

    def encode_video(self, video, encode_chunk_size=64):
        raise RuntimeError("vit_patch_hf should replace this method")


def postprocess_hook(video_features, **kwargs):
    assert video_features.ndim == 3
    assert kwargs["frames"] >= 1
    return video_features[:, :2, :]


def parse_args():
    parser = argparse.ArgumentParser(description="Certify the local ViT sparse patch with a dummy model.")
    parser.add_argument(
        "--result-path",
        default=str(ROOT / "results" / "vit_sparse_certification.json"),
        help="Path to write the JSON certification result.",
    )
    parser.add_argument(
        "--experiment-name",
        default="vit_sparse_certification",
        help="Experiment name stored in the JSON result.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    model = DummyModel()
    vit_patch_hf(
        model,
        cache_interval=2,
        update_token_ratio=0.5,
        vit_sparse_encode_chunk_size=1,
        vit_output_postprocess=postprocess_hook,
    )

    video = torch.zeros(3, 2, 2, 3)
    model.encode_video(video)

    layer = model.vision_tower.vision_model.encoder.layers[0]
    checks = {
        "patched_encode_video": hasattr(model, "_original_encode_video"),
        "patched_get_video_features": hasattr(model, "_original_get_video_features"),
        "final_chunk_idx": model.inference_context.chunk_idx == 2,
        "encoded_chunk_count": len(model.encoded_chunks) == 3,
        "postprocess_shape": all(shape == torch.Size([1, 2, 8]) for shape in model.encoded_chunks),
        "dense_reference_cache_created": hasattr(layer, "ref_k"),
        "sparse_forward_completed": True,
    }
    passed = all(checks.values())
    result = {
        "experiment_name": args.experiment_name,
        "passed": passed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "encoded_chunk_shapes": [list(shape) for shape in model.encoded_chunks],
        "checks": checks,
    }

    result_path = pathlib.Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not passed:
        raise AssertionError(json.dumps(result, indent=2))
    print("vit sparse patch verification passed")
    print(f"result written to {result_path}")


if __name__ == "__main__":
    main()
