from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class HFSigLIPBlockAdapter:
    def __init__(self, block: nn.Module):
        self.block = block
        self.norm1 = block.layer_norm1
        self.attn = block.self_attn
        self.norm2 = block.layer_norm2
        self.mlp = block.mlp

    def _project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.attn.q_proj(hidden_states)
        key = self.attn.k_proj(hidden_states)
        value = self.attn.v_proj(hidden_states)
        return query, key, value

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        head_dim = embed_dim // self.attn.num_heads
        return tensor.view(batch_size, seq_len, self.attn.num_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attn.embed_dim)

    def _attention_from_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        q_heads = self._split_heads(query)
        k_heads = self._split_heads(key)
        v_heads = self._split_heads(value)
        attn_output = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            dropout_p=self.attn.dropout if self.block.training else 0.0,
            is_causal=False,
        )
        return self.attn.out_proj(self._merge_heads(attn_output))

    def forward_with_cache(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = hidden_states
        normed = self.norm1(hidden_states)
        query, key, value = self._project_qkv(normed)
        attn_out = self._attention_from_qkv(query, key, value)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        cache = {
            "key": key.detach(),
            "value": value.detach(),
            "output": hidden_states.detach(),
        }
        return hidden_states, cache


class HFSigLIPVisionWrapper(nn.Module):
    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        self.model = vision_tower
        self.image_size = self.model.config.image_size
        self.blocks = [HFSigLIPBlockAdapter(block) for block in self.model.vision_model.encoder.layers]
        self.norm = self.model.vision_model.post_layernorm

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(device=self.device, dtype=self.dtype)
        return self.model.vision_model.embeddings(frames)

    def forward_with_layers(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = self.embed(frames)
        layer_outputs = []
        for block in self.blocks:
            hidden_states, _ = block.forward_with_cache(hidden_states)
            layer_outputs.append(hidden_states)
        return self.norm(hidden_states), layer_outputs

    def forward_with_caches(self, frames: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        hidden_states = self.embed(frames)
        caches = []
        for block in self.blocks:
            hidden_states, cache = block.forward_with_cache(hidden_states)
            caches.append(cache)
        return self.norm(hidden_states), caches
