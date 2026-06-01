import torch
import types
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
        model.semantic_stream_gate = SemanticStreamGate(
            refresh_interval=kwargs.get("semantic_refresh_interval", cache_interval),
            skip_patch_threshold=kwargs.get("semantic_skip_threshold", 0.01),
        )
        if not model.semantic_stream_compute_gate:
            model.vit_output_postprocess = model.semantic_stream_gate
    else:
        model.semantic_stream_compute_gate = False
        model.semantic_stream_gate = None

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

    kept_pixel_values = pixel_values_videos.index_select(1, keep_indices)
    video_features = self._get_video_features(kept_pixel_values)
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


@torch.inference_mode()
def _new_encode_video(self, video, encode_chunk_size=None):
    encode_chunk_size = encode_chunk_size or self.vit_sparse_encode_chunk_size
    num_frames = video.shape[0]
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
