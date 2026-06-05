import torch
import numpy as np
import time
import json
import math
from pathlib import Path
from logzero import logger
from decord import VideoReader, cpu
import imageio.v3 as iio

from video_qa.base import BaseVQA, work


class ReKVStreamVQA(BaseVQA):
    @staticmethod
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def load_video(self, video_path):
        video_path_obj = Path(video_path)
        if video_path_obj.is_dir():
            frame_paths = sorted(
                [
                    path
                    for path in video_path_obj.rglob("*")
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            )
            assert frame_paths, f"No image frames found in {video_path}"
            assert self.sample_fps <= 1
            step = max(1, round(1 / self.sample_fps))
            sampled_paths = frame_paths[::step]
            return np.stack([iio.imread(path) for path in sampled_paths])

        if video_path.endswith('.npy'):  # FPS=1
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps())
            if not math.isfinite(fps) or fps <= 0:
                fps = max(float(self.sample_fps), 1.0)
            step = max(1, round(fps / self.sample_fps))
            frame_idx = list(range(0, len(vr), step))
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(
            input_text,
            max_new_tokens=max_new_tokens,
            retrieved_indices=retrieved_indices,
        )

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def _is_latest_query(self, question):
        terms = getattr(self, "latest_query_terms", [])
        question = question.lower()
        return any(term in question for term in terms)

    def _retrievable_block_count(self):
        cache = getattr(self.qa_model, "kv_cache", None)
        if not cache:
            return 0
        layer_kv = cache[0]
        if getattr(layer_kv, "init_exc", False):
            return int(getattr(layer_kv, "num_global_block", 0))
        global_remainder = getattr(layer_kv, "global_remainder", None)
        if global_remainder is None:
            return 0
        token_count = int(global_remainder[0].size(-2)) - int(getattr(layer_kv, "n_init", 0))
        block_size = int(getattr(layer_kv, "block_size", self.qa_model.n_frame_tokens))
        return max(0, token_count // block_size)

    def _recent_retrieved_indices(self, keep_blocks):
        cache = getattr(self.qa_model, "kv_cache", None)
        if not cache or keep_blocks <= 0:
            return None, 0, 0
        layer_kv = cache[0]
        total_blocks = self._retrievable_block_count()
        if total_blocks <= 0:
            return None, total_blocks, 0
        kept_blocks = min(int(keep_blocks), total_blocks)
        start_idx = total_blocks - kept_blocks
        indices = list(range(start_idx, total_blocks))
        num_units = int(getattr(layer_kv, "num_units", 1))
        return [indices[:] for _ in range(num_units)], total_blocks, kept_blocks

    def _build_query_aware_retrieval(self, question):
        if not getattr(self, "enable_query_aware_retrieval", False):
            return None, "internal", self._retrievable_block_count(), 0
        policy = getattr(self, "query_retrieval_policy", "latest_recent")
        if policy == "internal":
            return None, "internal", self._retrievable_block_count(), 0
        if policy == "latest_recent" and not self._is_latest_query(question):
            return None, "internal", self._retrievable_block_count(), 0
        if policy not in {"latest_recent", "always_recent"}:
            raise ValueError(f"Unknown query_retrieval_policy: {policy}")
        retrieved_indices, total_blocks, kept_blocks = self._recent_retrieved_indices(
            getattr(self, "latest_retrieval_blocks", 0)
        )
        if retrieved_indices is None:
            return None, "internal", total_blocks, 0
        return retrieved_indices, policy, total_blocks, kept_blocks

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video_timer_start = time.perf_counter()
        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0
        load_start = time.perf_counter()
        video = self.load_video(video_path)
        load_sec = time.perf_counter() - load_start
        video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self._sync_cuda()
        init_start = time.perf_counter()
        self.qa_model.encode_init_prompt()
        self._sync_cuda()
        init_prompt_sec = time.perf_counter() - init_start
        cumulative_encode_video_sec = 0.0

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            temporal_windows = temporal_windows.tolist()

            # encode video until receiving QA
            new_frames = 0
            encode_video_sec = 0.0
            requested_end_idx = temporal_windows[-1]
            if requested_end_idx > video_start_idx:
                encode_start_idx = int(video_start_idx)
                encode_end_idx = min(len(video_tensor), math.ceil(requested_end_idx))
                video_end_idx = encode_end_idx
                new_frames = max(0, encode_end_idx - encode_start_idx)
                self._sync_cuda()
                encode_start = time.perf_counter()
                semantic_gate = getattr(self.qa_model, "semantic_stream_gate", None)
                if semantic_gate is not None and hasattr(semantic_gate, "set_recency_window"):
                    semantic_gate.set_recency_window(encode_start_idx, encode_end_idx)
                self.qa_model.encode_video(video_tensor[encode_start_idx:encode_end_idx])
                self._sync_cuda()
                encode_video_sec = time.perf_counter() - encode_start
                cumulative_encode_video_sec += encode_video_sec
                video_start_idx = encode_end_idx

            # 在 QA 解码前读取视频 KV cache（键值缓存）；解码过程可能释放或替换主缓存。
            kv_cache_memory_bytes = (
                int(self.qa_model.calc_memory_usage())
                if self.qa_model.kv_cache is not None
                else 0
            )

            # OpenQA
            self._sync_cuda()
            qa_start = time.perf_counter()
            retrieved_indices, query_route, retrievable_blocks, qa_retrieved_blocks = (
                self._build_query_aware_retrieval(question)
            )
            qa_results = self.video_open_qa(
                question,
                max_new_tokens=256,
                retrieved_indices=retrieved_indices,
            )
            self._sync_cuda()
            qa_sec = time.perf_counter() - qa_start
            semantic_gate = getattr(self.qa_model, "semantic_stream_gate", None)
            semantic_stats = getattr(semantic_gate, "stats", {}) if semantic_gate is not None else {}
            output_postprocess = getattr(
                self.qa_model,
                "vit_output_postprocess",
                None,
            )
            token_reducer_stats = getattr(output_postprocess, "stats", {})
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'benchmark': video_sample.get('benchmark', ''),
                'benchmark_group': video_sample.get('benchmark_group', ''),
                'benchmark_task': video_sample.get('benchmark_task', ''),
                'official_id': video_sample.get('official_id', ''),
                'query_index': video_sample.get('query_index', ''),
                'question': question,
                'answer': answer,
                'answer_type': sample.get('answer_type', ''),
                'eval_all': json.dumps(sample.get('eval_all', []), ensure_ascii=False),
                'eval_any': json.dumps(sample.get('eval_any', []), ensure_ascii=False),
                'eval_not': json.dumps(sample.get('eval_not', []), ensure_ascii=False),
                'pred_answer': qa_results['pred_answer'],
                'sample_fps': self.sample_fps,
                'loaded_frames': int(len(video)),
                'new_encoded_frames': int(new_frames),
                'encoded_until_frame': int(video_end_idx),
                'load_video_sec': load_sec,
                'init_prompt_sec': init_prompt_sec,
                'encode_video_sec': encode_video_sec,
                'cumulative_encode_video_sec': cumulative_encode_video_sec,
                'qa_sec': qa_sec,
                'elapsed_video_sec': time.perf_counter() - video_timer_start,
                'query_route': query_route,
                'retrievable_blocks': retrievable_blocks,
                'qa_retrieved_blocks': qa_retrieved_blocks,
                'semantic_input_frames': semantic_stats.get("input_frames", 0),
                'semantic_kept_frames': semantic_stats.get("kept_frames", 0),
                'semantic_skipped_frames': semantic_stats.get("skipped_frames", 0),
                'semantic_candidate_frames': semantic_stats.get("candidate_frames", 0),
                'semantic_preprocessed_frames': semantic_stats.get("preprocessed_frames", 0),
                'semantic_recency_kept_frames': semantic_stats.get("recency_kept_frames", 0),
                'semantic_coverage_kept_frames': semantic_stats.get("coverage_kept_frames", 0),
                'semantic_budget_kept_frames': semantic_stats.get("budget_kept_frames", 0),
                'semantic_reallocated_frames': semantic_stats.get(
                    "semantic_reallocated_frames",
                    0,
                ),
                'semantic_pair_rejected_frames': semantic_stats.get(
                    "semantic_pair_rejected_frames",
                    0,
                ),
                'semantic_input_tokens': semantic_stats.get("input_tokens", 0),
                'semantic_written_tokens': semantic_stats.get("written_tokens", 0),
                'semantic_raw_signature_mode': getattr(
                    self.qa_model,
                    "semantic_raw_signature_mode",
                    "",
                ),
                'semantic_raw_grid_size': getattr(
                    self.qa_model,
                    "semantic_raw_grid_size",
                    0,
                ),
                'semantic_raw_proposal_policy': getattr(
                    self.qa_model,
                    "semantic_raw_proposal_policy",
                    "",
                ),
                'semantic_saliency_z_threshold': getattr(
                    self.qa_model,
                    "semantic_saliency_z_threshold",
                    0.0,
                ),
                'semantic_pair_similarity_threshold': getattr(
                    self.qa_model,
                    "semantic_pair_similarity_threshold",
                    0.0,
                ),
                'semantic_proposal_sec': semantic_stats.get("proposal_sec", 0.0),
                'semantic_preprocess_sec': semantic_stats.get("preprocess_sec", 0.0),
                'semantic_embedding_sec': semantic_stats.get("embedding_sec", 0.0),
                'semantic_verification_sec': semantic_stats.get("verification_sec", 0.0),
                'semantic_vit_encoder_sec': semantic_stats.get("vit_encoder_sec", 0.0),
                'semantic_context_write_sec': semantic_stats.get("context_write_sec", 0.0),
                'vit_dense_frames': semantic_stats.get("vit_dense_frames", 0),
                'vit_sparse_frames': semantic_stats.get("vit_sparse_frames", 0),
                'vit_dense_sec': semantic_stats.get("vit_dense_sec", 0.0),
                'vit_sparse_sec': semantic_stats.get("vit_sparse_sec", 0.0),
                'vit_total_patch_tokens': semantic_stats.get(
                    "vit_total_patch_tokens",
                    0,
                ),
                'vit_updated_patch_tokens': semantic_stats.get(
                    "vit_updated_patch_tokens",
                    0,
                ),
                'vit_output_input_tokens': token_reducer_stats.get(
                    "input_tokens",
                    0,
                ),
                'vit_output_tokens': token_reducer_stats.get(
                    "output_tokens",
                    0,
                ),
                'vit_output_coverage_tokens': token_reducer_stats.get(
                    "coverage_tokens",
                    0,
                ),
                'vit_output_innovation_tokens': token_reducer_stats.get(
                    "innovation_tokens",
                    0,
                ),
                'kv_cache_memory_bytes': kv_cache_memory_bytes,
            })
 

if __name__ == "__main__":
    work(ReKVStreamVQA)
