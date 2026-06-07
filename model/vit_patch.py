import torch
import types
import time
from logzero import logger

from model.vision_accelerator import InferenceContext
from model.vision_accelerator import SemanticStreamGate
from model.vision_accelerator import FixedBudgetTokenReducer
from model.vision_accelerator import StructuredGridTokenReducer
from model.vision_accelerator import forward_siglip_adaptive
from model.vision_accelerator import new_siglip_sdpa_attn_forward


def vit_patch_hf(model, **kwargs):
    cache_interval = kwargs.get("cache_interval", 2)
    update_token_ratio = kwargs.get("update_token_ratio", 0.25)
    model.inference_context = InferenceContext(
        cache_interval=cache_interval,
        update_token_ratio=update_token_ratio,
    )
    model.vit_sparse_encode_chunk_size = kwargs.get("vit_sparse_encode_chunk_size", 1)
    output_postprocess = kwargs.get("vit_output_postprocess")
    output_token_policy = kwargs.get("vit_output_token_policy", "none")
    model.vit_output_selection_space = kwargs.get(
        "vit_output_selection_space",
        "projected",
    )
    model.vit_output_reduction_stage = "none"
    if output_postprocess is not None:
        model.vit_output_postprocess = output_postprocess
    elif output_token_policy == "structured_pool":
        model.vit_output_postprocess = StructuredGridTokenReducer(
            output_token_budget=int(
                kwargs.get("vit_output_token_budget", model.n_frame_tokens)
            ),
            reference_input_tokens=int(
                kwargs.get("vit_output_reference_tokens", 196)
            ),
        )
        model.vit_output_reduction_stage = "pre_projector"
    elif output_token_policy != "none":
        model.vit_output_postprocess = FixedBudgetTokenReducer(
            output_token_budget=int(
                kwargs.get("vit_output_token_budget", model.n_frame_tokens)
            ),
            coverage_tokens=int(
                kwargs.get("vit_output_coverage_tokens", 16)
            ),
            policy=output_token_policy,
            drift_feature_dims=int(
                kwargs.get("vit_output_drift_dims", 0)
            ),
        )
    else:
        model.vit_output_postprocess = _identity_vit_output_postprocess
    if kwargs.get("enable_semantic_stream", False):
        model.semantic_stream_compute_gate = kwargs.get("enable_semantic_compute_gate", False)
        model.semantic_selection_feature_source = kwargs.get("semantic_selection_feature_source", "vit_embedding")
        model.semantic_candidate_multiplier = kwargs.get("semantic_candidate_multiplier", 4)
        model.semantic_raw_signature_mode = kwargs.get("semantic_raw_signature_mode", "avg_pool")
        model.semantic_raw_grid_size = kwargs.get("semantic_raw_grid_size", 4)
        model.semantic_raw_proposal_policy = kwargs.get(
            "semantic_raw_proposal_policy",
            "novelty_topk",
        )
        model.semantic_saliency_z_threshold = kwargs.get(
            "semantic_saliency_z_threshold",
            4.0,
        )
        model.semantic_pair_similarity_threshold = kwargs.get(
            "semantic_pair_similarity_threshold",
            0.8,
        )
        model.semantic_profile_breakdown = kwargs.get("semantic_profile_breakdown", False)
        model.semantic_stream_gate = SemanticStreamGate(
            refresh_interval=kwargs.get("semantic_refresh_interval", cache_interval),
            skip_patch_threshold=kwargs.get("semantic_skip_threshold", 0.01),
            recency_keep_frames=kwargs.get("semantic_recency_keep_frames", 0),
            recency_updates_anchor=kwargs.get("semantic_recency_updates_anchor", False),
            coverage_interval=kwargs.get("semantic_coverage_interval", 0),
            coverage_updates_anchor=kwargs.get("semantic_coverage_updates_anchor", False),
            selection_policy=kwargs.get("semantic_selection_policy", "threshold"),
            budget_window_size=kwargs.get("semantic_budget_window_size", 0),
            budget_keep_per_window=kwargs.get("semantic_budget_keep_per_window", 1),
        )
        if not model.semantic_stream_compute_gate:
            model.vit_output_postprocess = model.semantic_stream_gate
    else:
        model.semantic_stream_compute_gate = False
        model.semantic_stream_gate = None

    model.enable_vit_layer_sparse = kwargs.get("enable_vit_layer_sparse", True)
    if model.enable_vit_layer_sparse:
        _apply_siglip_acceleration(model.vision_tower, model.inference_context)

    if hasattr(model, "_get_video_features") and not hasattr(model, "_original_get_video_features"):
        model._original_get_video_features = model._get_video_features
    model._get_video_features = types.MethodType(_new_get_video_features, model)

    if hasattr(model, "encode_video") and not hasattr(model, "_original_encode_video"):
        model._original_encode_video = model.encode_video
    model.encode_video = types.MethodType(_new_encode_video, model)

    logger.info("Vision tower sparse update patched successfully.")
    return model


def _apply_siglip_acceleration(vision_tower, context):
    for layer in vision_tower.vision_model.encoder.layers:
        layer._inference_context = context
        if not hasattr(layer, "_original_forward"):
            layer._original_forward = layer.forward
        layer.forward = types.MethodType(forward_siglip_adaptive, layer)
        layer.new_attn = types.MethodType(new_siglip_sdpa_attn_forward, layer)


def _identity_vit_output_postprocess(video_features, **kwargs):
    return video_features


def _profile_call(self, stat_key, function):
    if not getattr(self, "semantic_profile_breakdown", False):
        return function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    self.semantic_stream_gate.stats[stat_key] += time.perf_counter() - start
    return result


def _postprocess_vit_output(self, video_features, **kwargs):
    postprocess = getattr(self, "vit_output_postprocess", _identity_vit_output_postprocess)
    selected_video_feature = kwargs.get("selected_video_feature")
    if (
        getattr(self, "vit_output_selection_space", "projected") == "vit_native"
        and isinstance(postprocess, FixedBudgetTokenReducer)
        and selected_video_feature is not None
    ):
        # 在较低维的 ViT 原生空间判断变化，但仍从原投影输出中取 token，避免改变写入表示。
        kwargs["selection_features"] = self.apply_pooling(selected_video_feature)
    return postprocess(video_features, **kwargs)


def _project_and_postprocess_vit_output(
    self,
    selected_video_feature,
    batch_size,
    frames,
    **kwargs,
):
    postprocess = getattr(
        self,
        "vit_output_postprocess",
        _identity_vit_output_postprocess,
    )
    if isinstance(postprocess, StructuredGridTokenReducer):
        # 先压缩规则 ViT 网格，再执行 projector（投影器），同步减少视觉编码计算。
        reduced_feature = postprocess(
            selected_video_feature,
            batch_size=batch_size,
            frames=frames,
            **kwargs,
        )
        return self.multi_modal_projector(reduced_feature)
    projected = self.multi_modal_projector(selected_video_feature)
    pooled = self.apply_pooling(projected)
    return _postprocess_vit_output(
        self,
        pooled,
        batch_size=batch_size,
        frames=frames,
        selected_video_feature=selected_video_feature,
        **kwargs,
    )


def _new_get_video_features(self, pixel_values_videos):
    batch_size, frames, channels, height, width = pixel_values_videos.shape
    pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)

    video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
    selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

    if self.config.vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]
    elif self.config.vision_feature_select_strategy == "full":
        selected_video_feature = selected_video_feature

    video_features = _project_and_postprocess_vit_output(
        self,
        selected_video_feature,
        batch_size=batch_size,
        frames=frames,
        pixel_values_videos=pixel_values_videos,
    )
    if video_features.shape[0] == batch_size:
        return video_features
    video_features = video_features.reshape(batch_size, -1, video_features.shape[-1])
    return video_features


def _get_video_features_from_embeddings(self, embeddings, batch_size, frames):
    encoder_outputs = self.vision_tower.vision_model.encoder(
        inputs_embeds=embeddings,
        output_hidden_states=True,
    )
    selected_video_feature = encoder_outputs.hidden_states[self.config.vision_feature_layer]

    if self.config.vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]
    elif self.config.vision_feature_select_strategy == "full":
        selected_video_feature = selected_video_feature

    video_features = _project_and_postprocess_vit_output(
        self,
        selected_video_feature,
        batch_size=batch_size,
        frames=frames,
    )
    return video_features.reshape(batch_size, frames * video_features.shape[1], -1)


def _get_video_features_from_embeddings_streaming(self, embeddings):
    frame_features = []
    token_count = int(embeddings.shape[1])
    update_ratio = float(self.inference_context.update_token_ratio)
    semantic_stats = self.semantic_stream_gate.stats

    for frame_embedding in embeddings.split(1, dim=0):
        self.inference_context.step()
        is_reference = self.inference_context.is_reference_chunk
        if torch.cuda.is_available() and getattr(
            self,
            "semantic_profile_breakdown",
            False,
        ):
            torch.cuda.synchronize()
        start = time.perf_counter()
        encoder_outputs = self.vision_tower.vision_model.encoder(
            inputs_embeds=frame_embedding,
            output_hidden_states=True,
        )
        if torch.cuda.is_available() and getattr(
            self,
            "semantic_profile_breakdown",
            False,
        ):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        semantic_stats["vit_total_patch_tokens"] += token_count
        if is_reference:
            semantic_stats["vit_dense_frames"] += 1
            semantic_stats["vit_dense_sec"] += elapsed
            semantic_stats["vit_updated_patch_tokens"] += token_count
        else:
            # 该值表示算法计划更新的 token 数，不等同于融合算子后的实际 FLOPs。
            updated_tokens = max(1, int(token_count * update_ratio))
            semantic_stats["vit_sparse_frames"] += 1
            semantic_stats["vit_sparse_sec"] += elapsed
            semantic_stats["vit_updated_patch_tokens"] += updated_tokens

        selected_video_feature = encoder_outputs.hidden_states[
            self.config.vision_feature_layer
        ]
        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature

        frame_features.append(
            _project_and_postprocess_vit_output(
                self,
                selected_video_feature,
                batch_size=1,
                frames=1,
            )
        )

    if not frame_features:
        return embeddings.new_empty((1, 0, 0))
    return torch.cat(frame_features, dim=1)


def _encode_video_chunk_with_semantic_compute_gate(self, video_chunk):
    pixel_values_videos = self.processor.video_processor(
        video_chunk,
        return_tensors="pt",
    ).pixel_values_videos.to(self.device, self.dtype)

    batch_size, frames, channels, height, width = pixel_values_videos.shape
    if batch_size != 1:
        return self._encode_video_chunk(video_chunk)

    flat_pixels = pixel_values_videos.view(batch_size * frames, channels, height, width)
    embeddings = self.vision_tower.vision_model.embeddings(flat_pixels)
    signatures = self.semantic_stream_gate._frame_signature(embeddings)
    keep_indices = self.semantic_stream_gate.select_indices_from_signatures(
        signatures,
        token_count=self.n_frame_tokens,
    )
    if keep_indices.numel() == 0:
        return

    kept_embeddings = embeddings.index_select(0, keep_indices)
    video_features = _get_video_features_from_embeddings(
        self,
        kept_embeddings,
        batch_size=batch_size,
        frames=int(keep_indices.numel()),
    )
    if video_features.shape[1] == 0:
        return
    assert self.n_local >= video_features.shape[1], (
        f"n_local: {self.n_local}, video_features: {video_features.shape[1]}"
    )
    output = self.language_model(
        inputs_embeds=video_features,
        past_key_values=self.kv_cache,
        use_cache=True,
        return_dict=True,
    )
    self.kv_cache = output.past_key_values


def _encode_video_window_with_semantic_compute_gate(self, video):
    if video.shape[0] == 0:
        return
    feature_source = getattr(self, "semantic_selection_feature_source", "vit_embedding")
    selected_by_periodic = self.semantic_stream_gate.selection_policy == "periodic"
    selected_by_raw = feature_source == "raw_rgb"
    selected_by_hybrid = feature_source == "hybrid"
    if selected_by_periodic:
        keep_indices = _profile_call(
            self,
            "proposal_sec",
            lambda: self.semantic_stream_gate.select_periodic_indices(
                total_frames=int(video.shape[0]),
                token_count=self.n_frame_tokens,
                device=video.device,
            ),
        )
        if keep_indices.numel() == 0:
            return
        video = video.index_select(0, keep_indices)
    elif selected_by_raw:
        def select_raw_frames():
            signatures = _raw_rgb_signatures(
                video,
                grid_size=int(getattr(self, "semantic_raw_grid_size", 4)),
                mode=getattr(self, "semantic_raw_signature_mode", "avg_pool"),
            )
            indices = self.semantic_stream_gate.select_indices_from_signatures(
                signatures,
                token_count=self.n_frame_tokens,
            )
            return indices

        keep_indices = _profile_call(
            self,
            "proposal_sec",
            select_raw_frames,
        )
        if keep_indices.numel() == 0:
            return
        video = video.index_select(0, keep_indices.to(video.device if video.is_cuda else "cpu"))
    elif selected_by_hybrid:
        proposal_policy = getattr(
            self,
            "semantic_raw_proposal_policy",
            "novelty_topk",
        )

        def propose_candidates():
            signatures = _raw_rgb_signatures(
                video,
                grid_size=int(getattr(self, "semantic_raw_grid_size", 4)),
                mode=getattr(self, "semantic_raw_signature_mode", "avg_pool"),
            )
            indices = _raw_rgb_candidate_indices(
                signatures,
                self.semantic_stream_gate,
                int(getattr(self, "semantic_candidate_multiplier", 4)),
                proposal_policy=proposal_policy,
                saliency_z_threshold=float(
                    getattr(self, "semantic_saliency_z_threshold", 4.0)
                ),
            )
            return signatures, indices

        raw_signatures, candidate_indices = _profile_call(
            self,
            "proposal_sec",
            propose_candidates,
        )
        if candidate_indices.numel() == 0:
            return
        self.semantic_stream_gate.stats["candidate_frames"] += int(candidate_indices.numel())
        video = video.index_select(0, candidate_indices.to(video.device if video.is_cuda else "cpu"))

    self.semantic_stream_gate.stats["preprocessed_frames"] += int(video.shape[0])
    pixel_values_videos = _profile_call(
        self,
        "preprocess_sec",
        lambda: self.processor.video_processor(
            video,
            return_tensors="pt",
        ).pixel_values_videos.to(self.device, self.dtype),
    )

    batch_size, frames, channels, height, width = pixel_values_videos.shape
    if batch_size != 1:
        return self._encode_video_chunk(video)

    flat_pixels = pixel_values_videos.view(batch_size * frames, channels, height, width)
    embeddings = _profile_call(
        self,
        "embedding_sec",
        lambda: self.vision_tower.vision_model.embeddings(flat_pixels),
    )
    if selected_by_periodic or selected_by_raw:
        kept_embeddings = embeddings
    elif selected_by_hybrid:
        def verify_candidates():
            signatures = self.semantic_stream_gate._frame_signature(embeddings)
            if proposal_policy == "saliency_paired":
                # 同窗口比较周期帧和事件帧，只在语义差异足够大时重分配固定槽位。
                indices = self.semantic_stream_gate.select_indices_from_paired_candidate_signatures(
                    signatures,
                    candidate_indices.to(signatures.device),
                    total_frames=int(raw_signatures.shape[0]),
                    token_count=self.n_frame_tokens,
                    similarity_threshold=float(
                        getattr(self, "semantic_pair_similarity_threshold", 0.8)
                    ),
                )
            else:
                indices = self.semantic_stream_gate.select_indices_from_candidate_signatures(
                    signatures,
                    candidate_indices.to(signatures.device),
                    total_frames=int(raw_signatures.shape[0]),
                    token_count=self.n_frame_tokens,
                )
            return indices

        keep_indices = _profile_call(
            self,
            "verification_sec",
            verify_candidates,
        )
        if keep_indices.numel() == 0:
            return
        kept_embeddings = embeddings.index_select(0, keep_indices)
    else:
        def select_from_embeddings():
            signatures = self.semantic_stream_gate._frame_signature(embeddings)
            return self.semantic_stream_gate.select_indices_from_signatures(
                signatures,
                token_count=self.n_frame_tokens,
            )

        keep_indices = _profile_call(
            self,
            "verification_sec",
            select_from_embeddings,
        )
        if keep_indices.numel() == 0:
            return
        kept_embeddings = embeddings.index_select(0, keep_indices)
    if getattr(self, "enable_vit_layer_sparse", False):
        # 层内缓存只描述单帧参考，必须按时间顺序逐帧推进，不能批量混合多个帧。
        encode_kept_embeddings = lambda: _get_video_features_from_embeddings_streaming(
            self,
            kept_embeddings,
        )
    else:
        encode_kept_embeddings = lambda: _get_video_features_from_embeddings(
            self,
            kept_embeddings,
            batch_size=batch_size,
            frames=int(keep_indices.numel()),
        )
    video_features = _profile_call(
        self,
        "vit_encoder_sec",
        encode_kept_embeddings,
    )
    if video_features.shape[1] == 0:
        return
    assert self.n_local >= video_features.shape[1], (
        f"n_local: {self.n_local}, video_features: {video_features.shape[1]}"
    )
    output = _profile_call(
        self,
        "context_write_sec",
        lambda: self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        ),
    )
    self.kv_cache = output.past_key_values


def _raw_rgb_signatures(video, grid_size=4, mode="avg_pool"):
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor [frames, height, width, channels], got {tuple(video.shape)}")
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    if mode == "avg_pool":
        frames = video.float()
        if video.dtype == torch.uint8:
            frames = frames / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()
        signatures = torch.nn.functional.adaptive_avg_pool2d(
            frames,
            (grid_size, grid_size),
        ).flatten(1)
    elif mode in {"grid_sample", "grid_sample_stable"}:
        height, width = int(video.shape[1]), int(video.shape[2])
        y_indices = (
            (torch.arange(grid_size, device=video.device, dtype=torch.float32) + 0.5)
            * height
            / grid_size
        ).long().clamp(max=height - 1)
        x_indices = (
            (torch.arange(grid_size, device=video.device, dtype=torch.float32) + 0.5)
            * width
            / grid_size
        ).long().clamp(max=width - 1)
        sampled = video.index_select(1, y_indices).index_select(2, x_indices).float()
        if video.dtype == torch.uint8:
            sampled = sampled / 255.0
        signatures = sampled.flatten(1)
        if mode == "grid_sample_stable":
            # 追加常数维度，避免纯黑帧形成零向量并被余弦距离误判为剧烈变化。
            signatures = torch.cat(
                [
                    signatures,
                    torch.ones(
                        signatures.shape[0],
                        1,
                        device=signatures.device,
                        dtype=signatures.dtype,
                    ),
                ],
                dim=-1,
            )
    else:
        raise ValueError(f"Unknown raw RGB signature mode: {mode}")
    return torch.nn.functional.normalize(signatures, dim=-1)


def _raw_rgb_candidate_indices(
    raw_signatures,
    semantic_gate,
    candidate_multiplier,
    proposal_policy="novelty_topk",
    saliency_z_threshold=4.0,
):
    if proposal_policy not in {
        "novelty_topk",
        "saliency_gated",
        "saliency_paired",
    }:
        raise ValueError(f"Unknown raw proposal policy: {proposal_policy}")
    candidate_multiplier = max(1, candidate_multiplier)
    total_frames = int(raw_signatures.shape[0])
    if total_frames == 0:
        return torch.empty(0, device=raw_signatures.device, dtype=torch.long)

    base_frame_idx = int(semantic_gate.frame_idx)
    budget_window_size = max(1, int(semantic_gate.budget_window_size))
    budget_keep_per_window = max(1, int(semantic_gate.budget_keep_per_window))
    candidate_budget = candidate_multiplier * budget_keep_per_window
    forced = set()
    reserved = set()
    if semantic_gate.anchor_feature is None:
        forced.add(0)
        reserved.add(0)
    for local_idx in range(total_frames):
        global_idx = base_frame_idx + local_idx
        if global_idx % int(semantic_gate.refresh_interval) == 0:
            forced.add(local_idx)
            reserved.add(local_idx)
        if semantic_gate.coverage_interval > 0 and global_idx % int(semantic_gate.coverage_interval) == 0:
            forced.add(local_idx)
            reserved.add(local_idx)
        if semantic_gate._in_recency_window(global_idx):
            forced.add(local_idx)

    prev = raw_signatures[:-1]
    curr = raw_signatures[1:]
    if curr.shape[0] == 0:
        deltas = [0.0]
    else:
        similarities = torch.nn.functional.cosine_similarity(curr, prev, dim=-1)
        deltas = [0.0] + [max(0.0, float(1.0 - value.item())) for value in similarities]

    if proposal_policy == "saliency_paired":
        selected = set()
        candidates_by_window = {}
        for local_idx, drift in enumerate(deltas):
            global_idx = base_frame_idx + local_idx
            if semantic_gate._in_recency_window(global_idx):
                selected.add(local_idx)
                continue
            window_id = global_idx // budget_window_size
            candidates_by_window.setdefault(window_id, []).append((drift, local_idx))

        for window_id, candidates in candidates_by_window.items():
            window_start_global = window_id * budget_window_size
            periodic_local_idx = window_start_global - base_frame_idx
            if not 0 <= periodic_local_idx < total_frames:
                continue
            selected.add(periodic_local_idx)
            if semantic_gate.anchor_feature is None and window_start_global == 0:
                # 首窗口建立全局参考锚点，不允许事件候选替换初始化覆盖。
                continue

            candidates.sort(reverse=True)
            top_drift, top_local_idx = candidates[0]
            window_values = torch.tensor(
                [drift for drift, _ in candidates],
                device=raw_signatures.device,
                dtype=torch.float32,
            )
            mean = float(window_values.mean().item())
            std = float(window_values.std(unbiased=False).item())
            z_score = (top_drift - mean) / (std + 1e-6)
            if (
                z_score >= saliency_z_threshold
                and top_local_idx != periodic_local_idx
            ):
                # RGB 只提出备选事件；最终是否替换由 ViT 嵌入层的配对语义差异决定。
                selected.add(top_local_idx)

        return torch.tensor(
            sorted(selected),
            device=raw_signatures.device,
            dtype=torch.long,
        )

    candidates_by_window = {}
    for local_idx, drift in enumerate(deltas):
        if local_idx in forced:
            continue
        global_idx = base_frame_idx + local_idx
        window_id = global_idx // budget_window_size
        candidates_by_window.setdefault(window_id, []).append((drift, local_idx))

    selected = set(forced)
    for window_id, candidates in candidates_by_window.items():
        if proposal_policy == "saliency_gated":
            reserved_count = sum(
                (base_frame_idx + local_idx) // budget_window_size == window_id
                for local_idx in reserved
            )
            if reserved_count >= budget_keep_per_window:
                continue
            candidates.sort(reverse=True)
            top_drift, top_local_idx = candidates[0]
            window_values = torch.tensor(
                [drift for drift, _ in candidates],
                device=raw_signatures.device,
                dtype=torch.float32,
            )
            mean = float(window_values.mean().item())
            std = float(window_values.std(unbiased=False).item())
            z_score = (top_drift - mean) / (std + 1e-6)
            window_start_global = window_id * budget_window_size
            periodic_local_idx = window_start_global - base_frame_idx
            if (
                z_score < saliency_z_threshold
                and 0 <= periodic_local_idx < total_frames
                and periodic_local_idx not in forced
            ):
                selected.add(periodic_local_idx)
            else:
                selected.add(top_local_idx)
            continue

        candidates.sort(reverse=True)
        for _, local_idx in candidates[:candidate_budget]:
            selected.add(local_idx)

    return torch.tensor(sorted(selected), device=raw_signatures.device, dtype=torch.long)


@torch.inference_mode()
def _new_encode_video(self, video, encode_chunk_size=None):
    encode_chunk_size = encode_chunk_size or self.vit_sparse_encode_chunk_size
    num_frames = video.shape[0]
    semantic_gate = getattr(self, "semantic_stream_gate", None)
    if (
        getattr(self, "semantic_stream_compute_gate", False)
        and getattr(semantic_gate, "selection_policy", "threshold")
        in {"budget_topk", "periodic"}
    ):
        _encode_video_window_with_semantic_compute_gate(self, video)
        return

    num_chunks = num_frames // encode_chunk_size

    for chunk_idx in range(num_chunks):
        self.inference_context.step()
        start_idx = chunk_idx * encode_chunk_size
        end_idx = start_idx + encode_chunk_size
        chunk_video = video[start_idx:end_idx]
        if getattr(self, "semantic_stream_compute_gate", False):
            _encode_video_chunk_with_semantic_compute_gate(self, chunk_video)
        else:
            self._encode_video_chunk(chunk_video)

    remaining_frames = num_frames % encode_chunk_size
    if remaining_frames > 0:
        self.inference_context.step()
        start_idx = num_chunks * encode_chunk_size
        end_idx = start_idx + remaining_frames
        remaining_video = video[start_idx:end_idx]
        if getattr(self, "semantic_stream_compute_gate", False):
            _encode_video_chunk_with_semantic_compute_gate(self, remaining_video)
        else:
            self._encode_video_chunk(remaining_video)
