import torch
import torch.nn.functional as F

# =========================================================================
#  入口函数 (将被 Patch 到 Layer.forward)
# =========================================================================

def forward_siglip_adaptive(self, 
                            hidden_states: torch.Tensor, 
                            attention_mask: torch.Tensor, 
                            output_attentions: bool = False):
    """
    自适应的 SigLIP Layer Forward 函数。
    根据 self._inference_context 的状态决定走全量计算还是稀疏计算。
    """
    # 获取绑定在 Layer 上的上下文 (在 patch_vision.py 中绑定的)
    ctx = self._inference_context
    if ctx.is_reference_chunk:
        return _forward_dense(self, 
                              hidden_states, 
                              attention_mask, 
                              output_attentions)
    else:
        return _forward_sparse(self, 
                               hidden_states, 
                               attention_mask, 
                               output_attentions, 
                               ctx.update_token_ratio)


# =========================================================================
#  核心实现：全量计算 (Reference)
# =========================================================================

def _forward_dense(self, hidden_states, attention_mask, output_attentions):
    """全量计算并保存 Reference Cache"""
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    
    # 1. Attention Projection
    batch_size, seq_len, embed_dim = hidden_states.shape
    num_heads = self.self_attn.num_heads
    head_dim = embed_dim // num_heads

    q = self.self_attn.q_proj(hidden_states)
    k = self.self_attn.k_proj(hidden_states)
    v = self.self_attn.v_proj(hidden_states)

    # [Cache] 保存最后一帧的 Projection 结果 (作为基准)
    self.ref_k = k[-1].detach().clone() 
    self.ref_v = v[-1].detach().clone()

    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    attn_output, attn_weights = self.new_attn(q, k, v, attention_mask, output_attentions)
    
    hidden_states = residual + attn_output
    
    # [Cache] 保存 Attn Output (用于后续恢复)
    self.ref_attn_out = attn_output[-1].detach().clone()

    # 2. MLP
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    mlp_output = self.mlp(hidden_states)
    
    # [Cache] 保存 MLP Output
    self.ref_mlp_out = mlp_output[-1].detach().clone()

    hidden_states = residual + mlp_output
    
    outputs = (hidden_states,)
    if output_attentions: outputs += (attn_weights,)
    return outputs


# =========================================================================
#  核心实现：稀疏计算 (Sparse Update)
# =========================================================================

def _forward_sparse(self, hidden_states, attention_mask, output_attentions, update_ratio):
    """基于 Key 相似度的部分更新"""
    residual = hidden_states
    hidden_states_ln1 = self.layer_norm1(hidden_states)
    
    batch_size, seq_len, embed_dim = hidden_states_ln1.shape
    num_heads = self.self_attn.num_heads
    head_dim = embed_dim // num_heads

    # 1. 计算完整的 Key (用于 Attention 也用于相似度比较)
    # 优化：只计算一次 k_proj
    key_states_full_flat = self.self_attn.k_proj(hidden_states_ln1) # [B, T, C]
    
    # 2. 计算相似度 & 选择 Indices
    # ref_k 是 [T, C]，unsqueeze 变成 [1, T, C] 进行广播
    # 计算当前帧与参考帧的 Cosine 相似度
    similarity = F.cosine_similarity(key_states_full_flat, self.ref_k.unsqueeze(0), dim=-1)
    
    num_update = max(1, int(seq_len * update_ratio))
    # 选出相似度最低的 (最不像的) Top-K
    update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices # [B, num_update]

    # 3. 准备 Sparse Attention 输入
    # 提取需要更新的 Token 特征
    update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
    tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded) # [B, num_update, C]

    # 只对这些 Token 算 Q 和 V
    q_selected = self.self_attn.q_proj(tokens_to_update)
    v_selected = self.self_attn.v_proj(tokens_to_update)

    # Reshape
    q_selected = q_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
    v_selected = v_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
    k_full = key_states_full_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # 4. 构造混合 V 矩阵 (Scatter Update)
    # 复制 Reference V 作为底板
    v_mixed = self.ref_v.unsqueeze(0).expand(batch_size, -1, -1).clone()
    v_mixed = v_mixed.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).clone()
    
    # 将新计算的 V 填入
    scatter_idx = update_indices.view(batch_size, 1, num_update, 1).expand(-1, num_heads, -1, head_dim)
    v_mixed.scatter_(2, scatter_idx, v_selected)

    # 5. 执行 Attention
    # 注意：这里输出的是选定 Token 的 Attention 结果，长度为 num_update
    attn_out_selected, _ = self.new_attn(q_selected, 
                                   k_full, 
                                   v_mixed, 
                                   attention_mask, 
                                   output_attentions)

    # 6. 构造混合 Attention Output
    attn_out_mixed = self.ref_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()
    # 将结果填回对应位置
    attn_out_mixed.scatter_(1, update_idx_expanded, attn_out_selected)

    hidden_states = residual + attn_out_mixed

    # 7. MLP 部分 (同样只计算选中的)
    residual_2 = hidden_states
    hidden_states_ln2 = self.layer_norm2(hidden_states)
    
    # 提取需要更新的 MLP 输入
    tokens_ln2_update = hidden_states_ln2.gather(1, update_idx_expanded)
    mlp_out_selected = self.mlp(tokens_ln2_update)
    
    # 混合 MLP 输出
    mlp_out_mixed = self.ref_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()
    mlp_out_mixed.scatter_(1, update_idx_expanded, mlp_out_selected)
    
    hidden_states = residual_2 + mlp_out_mixed

    return (hidden_states, None) if output_attentions else (hidden_states,)