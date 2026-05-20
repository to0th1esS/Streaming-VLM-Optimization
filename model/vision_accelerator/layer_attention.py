import torch
import torch.nn.functional
from typing import Optional, Tuple
from logzero import logger

def new_siglip_sdpa_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    q_len=query_states.shape[-2]
    batch_size=query_states.shape[0]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
        if not query_states.is_contiguous():
            query_states = query_states.contiguous()
        if not key_states.is_contiguous():
            key_states = key_states.contiguous()
        if not value_states.is_contiguous():
            value_states = value_states.contiguous()
    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling. 

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

    attn_output = self.self_attn.out_proj(attn_output)
    return attn_output, None