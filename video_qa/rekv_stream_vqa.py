import torch
import numpy as np
import time
import json
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
                    for path in video_path_obj.iterdir()
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
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

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
            if temporal_windows[-1] > video_end_idx:
                encode_start_idx = int(video_start_idx)
                video_end_idx = temporal_windows[-1]
                encode_end_idx = int(video_end_idx)
                new_frames = max(0, encode_end_idx - encode_start_idx)
                self._sync_cuda()
                encode_start = time.perf_counter()
                self.qa_model.encode_video(video_tensor[encode_start_idx:encode_end_idx])
                self._sync_cuda()
                encode_video_sec = time.perf_counter() - encode_start
                cumulative_encode_video_sec += encode_video_sec
                video_start_idx = video_end_idx
        
            # OpenQA
            self._sync_cuda()
            qa_start = time.perf_counter()
            qa_results = self.video_open_qa(question, max_new_tokens=256)
            self._sync_cuda()
            qa_sec = time.perf_counter() - qa_start
            semantic_gate = getattr(self.qa_model, "semantic_stream_gate", None)
            semantic_stats = getattr(semantic_gate, "stats", {}) if semantic_gate is not None else {}
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
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
                'semantic_input_frames': semantic_stats.get("input_frames", 0),
                'semantic_kept_frames': semantic_stats.get("kept_frames", 0),
                'semantic_skipped_frames': semantic_stats.get("skipped_frames", 0),
                'semantic_input_tokens': semantic_stats.get("input_tokens", 0),
                'semantic_written_tokens': semantic_stats.get("written_tokens", 0),
            })
 

if __name__ == "__main__":
    work(ReKVStreamVQA)
