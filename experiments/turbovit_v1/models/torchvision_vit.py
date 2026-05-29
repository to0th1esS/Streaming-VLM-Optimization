from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class TorchvisionViTBlockAdapter:
    def __init__(self, block: nn.Module):
        self.block = block
        self.norm1 = block.ln_1
        self.attn = block.self_attention
        self.norm2 = block.ln_2
        self.mlp = block.mlp
        self.dropout = block.dropout

    def _project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_dim = hidden_states.shape[-1]
        weight = self.attn.in_proj_weight
        bias = self.attn.in_proj_bias
        q = F.linear(hidden_states, weight[:embed_dim], bias[:embed_dim])
        key = F.linear(hidden_states, weight[embed_dim : 2 * embed_dim], bias[embed_dim : 2 * embed_dim])
        value = F.linear(hidden_states, weight[2 * embed_dim :], bias[2 * embed_dim :])
        return q, key, value

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        head_dim = embed_dim // self.attn.num_heads
        return tensor.view(batch_size, seq_len, self.attn.num_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attn.embed_dim)

    def forward_with_cache(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = hidden_states
        normed = self.norm1(hidden_states)
        _, key, _ = self._project_qkv(normed)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        hidden_states = residual + self.dropout(attn_out)
        mlp_out = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_out
        cache = {
            "key": key.detach(),
            "output": hidden_states.detach(),
        }
        return hidden_states, cache


class TorchvisionViTWrapper(nn.Module):
    def __init__(self, weights: str = "none"):
        super().__init__()
        if weights == "none":
            weight_enum = None
        elif weights == "imagenet":
            weight_enum = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"Unsupported weights: {weights}")
        self.model = vit_b_16(weights=weight_enum)
        self.image_size = self.model.image_size
        self.blocks = [TorchvisionViTBlockAdapter(block) for block in self.model.encoder.layers]
        self.norm = self.model.encoder.ln

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model._process_input(frames)
        batch_size = hidden_states.shape[0]
        class_token = self.model.class_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([class_token, hidden_states], dim=1)
        hidden_states = hidden_states + self.model.encoder.pos_embedding
        return self.model.encoder.dropout(hidden_states)

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        return self._embed(frames)

    def forward_with_layers(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self._embed(frames)
        layer_outputs = []
        for block in self.blocks:
            hidden_states, _ = block.forward_with_cache(hidden_states)
            layer_outputs.append(hidden_states)
        hidden_states = self.model.encoder.ln(hidden_states)
        return hidden_states, layer_outputs

    def forward_with_caches(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        hidden_states = self._embed(frames)
        caches = []
        for block in self.blocks:
            hidden_states, cache = block.forward_with_cache(hidden_states)
            caches.append(cache)
        hidden_states = self.model.encoder.ln(hidden_states)
        return hidden_states, caches
