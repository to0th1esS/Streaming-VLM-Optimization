from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPVisionModel


class HFCLIPBlockAdapter:
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

    def forward_with_cache(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = hidden_states
        normed = self.norm1(hidden_states)
        _, key, value = self._project_qkv(normed)
        attn_out, _ = self.attn(
            hidden_states=normed,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )
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


class HFCLIPVisionWrapper(nn.Module):
    def __init__(self, model_path: str, local_files_only: bool = True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.image_size = self.model.config.image_size
        self.blocks = [HFCLIPBlockAdapter(block) for block in self.model.vision_model.encoder.layers]
        self.norm = nn.Identity()

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.vision_model.embeddings(frames)
        return self.model.vision_model.pre_layrnorm(hidden_states)

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
