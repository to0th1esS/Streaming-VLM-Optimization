import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def clear_cache(self):
        self.kv_cache = None
        semantic_gate = getattr(self, "semantic_stream_gate", None)
        if semantic_gate is not None:
            semantic_gate.reset()
        inference_context = getattr(self, "inference_context", None)
        if inference_context is not None and hasattr(inference_context, "reset"):
            # 语言缓存和视觉参考必须在同一视频边界同时失效。
            inference_context.reset()
        output_postprocess = getattr(self, "vit_output_postprocess", None)
        if output_postprocess is not None and hasattr(output_postprocess, "reset"):
            # 输出 token 的 rolling anchor（滚动锚点）不能跨视频复用。
            output_postprocess.reset()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        if video_features.shape[1] == 0:
            return
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        # encode chunk by chunk
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory

    @staticmethod
    def _tensor_memory_bytes(tensor):
        if not isinstance(tensor, torch.Tensor):
            return 0
        return int(tensor.numel() * tensor.element_size())

    def calc_cache_memory_usage(self):
        """统计 ReKV 实际持有的 GPU/CPU KV cache（键值缓存）张量。"""
        if self.kv_cache is None:
            return {
                "cpu_bytes": 0,
                "gpu_bytes": 0,
                "total_bytes": 0,
                "logical_tokens": 0,
            }

        cpu_bytes = 0
        gpu_bytes = 0
        logical_tokens = 0
        for layer_cache in self.kv_cache:
            logical_tokens = max(
                logical_tokens,
                int(getattr(layer_cache, "length", 0)),
            )
            for name in ("local_k", "local_v", "init_k", "init_v", "global_buffer"):
                gpu_bytes += self._tensor_memory_bytes(
                    getattr(layer_cache, name, None)
                )
            for tensor in getattr(layer_cache, "global_remainder", ()):
                gpu_bytes += self._tensor_memory_bytes(tensor)

            cuda_cache = getattr(layer_cache, "cuda_cache", None)
            gpu_bytes += self._tensor_memory_bytes(
                getattr(cuda_cache, "data", None)
            )
            for block_keys in getattr(layer_cache, "block_k", ()):
                gpu_bytes += self._tensor_memory_bytes(
                    getattr(block_keys, "data", None)
                )

            # MemoryUnit 的 gpu_data 是 cuda_cache.data 的视图，不能重复计数。
            for unit_blocks in getattr(layer_cache, "global_blocks", ()):
                for block in unit_blocks:
                    for tensor in getattr(block, "cpu_data", ()):
                        cpu_bytes += self._tensor_memory_bytes(tensor)

        return {
            "cpu_bytes": cpu_bytes,
            "gpu_bytes": gpu_bytes,
            "total_bytes": cpu_bytes + gpu_bytes,
            "logical_tokens": logical_tokens,
        }
