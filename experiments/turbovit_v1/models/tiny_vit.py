from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TinyViTConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 96
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 2.0


class TinyViTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_cache(hidden_states)
        return output

    def _project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_dim = hidden_states.shape[-1]
        weight = self.attn.in_proj_weight
        bias = self.attn.in_proj_bias
        q = F.linear(hidden_states, weight[:embed_dim], bias[:embed_dim])
        k = F.linear(hidden_states, weight[embed_dim : 2 * embed_dim], bias[embed_dim : 2 * embed_dim])
        v = F.linear(hidden_states, weight[2 * embed_dim :], bias[2 * embed_dim :])
        return q, k, v

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
        hidden_states = residual + attn_out
        mlp_out = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_out
        cache = {
            "key": key.detach(),
            "output": hidden_states.detach(),
        }
        return hidden_states, cache

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        ref_key: torch.Tensor,
        ref_output: torch.Tensor,
        dynamic_ratio: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = hidden_states
        normed = self.norm1(hidden_states)
        q, key, value = self._project_qkv(normed)

        similarity = F.cosine_similarity(key, ref_key, dim=-1)
        seq_len = hidden_states.shape[1]
        num_dynamic = max(1, min(seq_len, int(round(seq_len * dynamic_ratio))))
        dynamic_indices = torch.topk(similarity, k=num_dynamic, dim=1, largest=False).indices
        gather_idx = dynamic_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])

        q_selected = q.gather(1, gather_idx)
        residual_selected = residual.gather(1, gather_idx)

        q_heads = self._split_heads(q_selected)
        k_heads = self._split_heads(key)
        v_heads = self._split_heads(value)
        scale = q_heads.shape[-1] ** -0.5
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_selected = torch.matmul(attn_probs, v_heads)
        attn_selected = self._merge_heads(attn_selected)
        attn_selected = self.attn.out_proj(attn_selected)

        hidden_selected = residual_selected + attn_selected
        mlp_selected = self.mlp(self.norm2(hidden_selected))
        output_selected = hidden_selected + mlp_selected

        output = ref_output.clone()
        output.scatter_(1, gather_idx, output_selected)
        cache = {
            "key": key.detach(),
            "output": output.detach(),
            "dynamic_indices": dynamic_indices.detach(),
            "similarity": similarity.detach(),
        }
        return output, cache


class TinyViTEncoder(nn.Module):
    def __init__(self, config: TinyViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        self.blocks = nn.ModuleList(
            TinyViTBlock(config.embed_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(frames)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states + self.pos_embed

    def forward_with_layers(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self.embed(frames)
        layer_outputs = []
        for block in self.blocks:
            hidden_states, _ = block.forward_with_cache(hidden_states)
            layer_outputs.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, layer_outputs

    def forward_with_caches(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        hidden_states = self.embed(frames)
        caches = []
        for block in self.blocks:
            hidden_states, cache = block.forward_with_cache(hidden_states)
            caches.append(cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states, caches
