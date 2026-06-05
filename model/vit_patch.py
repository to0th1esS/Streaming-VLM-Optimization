import torch
import types
import time
from logzero import logger

from model.vision_accelerator import InferenceContext
from model.vision_accelerator import SemanticStreamGate
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
    model.vit_output_postprocess = kwargs.get(
        "vit_output_postprocess",
        _identity_vit_output_postprocess,
    )
    if kwargs.get("enable_semantic_stream", False):
        model.semantic_stream_compute_gate = kwargs.get("enable_semantic_compute_gate", False)
        model.semantic_selection_feature_source = kwargs.get("semantic_selection_feature_source", "vit_embedding")
        model.semantic_candidate_multiplier = kwargs.get("semantic_candidate_multiplier", 4)
        model.semantic_raw_signature_mode = kwargs.get("semantic_raw_signature_mode", "avg_pool")
        model.semantic_raw_grid_size = kwargs.get("semantic_raw_grid_size", 4)
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
    return postprocess(video_features, **kwargs)


def _new_get_video_features(self, pixel_values_videos):
    batch_size, frames, channels, height, width = pixel_values_videos.shape
    pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)

    video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
    selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

    if self.config.vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]
    elif self.config.vision_feature_select_strategy == "full":
        selected_video_feature = selected_video_feature

    video_features = self.multi_modal_projector(selected_video_feature)
    video_features = self.apply_pooling(video_features)
    video_features = _postprocess_vit_output(
        self,
        video_features,
        batch_size=batch_size,
        frames=frames,
        pixel_values_videos=pixel_values_videos,
        selected_video_feature=selected_video_feature,
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

    video_features = self.multi_modal_projector(selected_video_feature)
    video_features = self.apply_pooling(video_features)
    return video_features.reshape(batch_size, frames * video_features.shape[1], -1)


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
    video_features = _profile_call(
        self,
        "vit_encoder_sec",
        lambda: _get_video_features_from_embeddings(
            self,
            kept_embeddings,
            batch_size=batch_size,
            frames=int(keep_indices.numel()),
        ),
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
    elif mode == "grid_sample":
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
    else:
        raise ValueError(f"Unknown raw RGB signature mode: {mode}")
    return torch.nn.functional.normalize(signatures, dim=-1)


def _raw_rgb_candidate_indices(raw_signatures, semantic_gate, candidate_multiplier):
    candidate_multiplier = max(1, candidate_multiplier)
    total_frames = int(raw_signatures.shape[0])
    if total_frames == 0:
        return torch.empty(0, device=raw_signatures.device, dtype=torch.long)

    base_frame_idx = int(semantic_gate.frame_idx)
    budget_window_size = max(1, int(semantic_gate.budget_window_size))
    budget_keep_per_window = max(1, int(semantic_gate.budget_keep_per_window))
    candidate_budget = candidate_multiplier * budget_keep_per_window
    forced = set()
    if semantic_gate.anchor_feature is None:
        forced.add(0)
    for local_idx in range(total_frames):
        global_idx = base_frame_idx + local_idx
        if global_idx % int(semantic_gate.refresh_interval) == 0:
            forced.add(local_idx)
        if semantic_gate.coverage_interval > 0 and global_idx % int(semantic_gate.coverage_interval) == 0:
            forced.add(local_idx)
        if semantic_gate._in_recency_window(global_idx):
            forced.add(local_idx)

    prev = raw_signatures[:-1]
    curr = raw_signatures[1:]
    if curr.shape[0] == 0:
        deltas = [0.0]
    else:
        similarities = torch.nn.functional.cosine_similarity(curr, prev, dim=-1)
        deltas = [0.0] + [max(0.0, float(1.0 - value.item())) for value in similarities]

    candidates_by_window = {}
    for local_idx, drift in enumerate(deltas):
        if local_idx in forced:
            continue
        global_idx = base_frame_idx + local_idx
        window_id = global_idx // budget_window_size
        candidates_by_window.setdefault(window_id, []).append((drift, local_idx))

    selected = set(forced)
    for candidates in candidates_by_window.values():
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
        self.inference_context.update(chunk_idx)
        start_idx = chunk_idx * encode_chunk_size
        end_idx = start_idx + encode_chunk_size
        chunk_video = video[start_idx:end_idx]
        if getattr(self, "semantic_stream_compute_gate", False):
            _encode_video_chunk_with_semantic_compute_gate(self, chunk_video)
        else:
            self._encode_video_chunk(chunk_video)

    remaining_frames = num_frames % encode_chunk_size
    if remaining_frames > 0:
        self.inference_context.update(num_chunks)
        start_idx = num_chunks * encode_chunk_size
        end_idx = start_idx + remaining_frames
        remaining_video = video[start_idx:end_idx]
        if getattr(self, "semantic_stream_compute_gate", False):
            _encode_video_chunk_with_semantic_compute_gate(self, remaining_video)
        else:
            self._encode_video_chunk(remaining_video)
