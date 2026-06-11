import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TASK_GROUPS = {
    "EPM": "backward",
    "ASI": "backward",
    "HLD": "backward",
    "OCR": "realtime",
    "ACR": "realtime",
    "ATR": "realtime",
    "STU": "realtime",
    "FPD": "realtime",
    "OJR": "realtime",
    "REC": "forward",
    "SSR": "forward",
    "CRR": "forward",
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def extract_choice(response):
    match = re.search(r"(?<![A-Z])([A-H])(?![A-Z])", response.upper())
    return match.group(1) if match else ""


def official_compatible_score(task, response, ground_truth):
    if not response:
        return 0
    if task in {"REC"}:
        digits = "".join(re.findall(r"\d+", response))
        return int(digits == str(ground_truth))
    if task in {"SSR", "CRR"}:
        if (response == "N" and ground_truth == "No") or (
            response == "Y" and ground_truth == "Yes"
        ):
            return 1
        return int(str(ground_truth) in response)
    return int(str(ground_truth) in response)


def strict_score(task, response, ground_truth):
    if task == "REC":
        numbers = re.findall(r"\d+", response)
        return int(len(numbers) == 1 and numbers[0] == str(ground_truth))
    if task in {"SSR", "CRR"}:
        normalized = response.strip().lower().rstrip(".")
        return int(normalized == str(ground_truth).lower())
    return int(extract_choice(response) == str(ground_truth))


def evaluate_rows(rows):
    evaluated = []
    for row in rows:
        task = row.get("benchmark_task", "")
        if task not in TASK_GROUPS:
            raise ValueError(f"Missing or unknown benchmark_task: {task}")
        response = row.get("pred_answer", "")
        ground_truth = row.get("answer", "")
        evaluated.append(
            {
                **row,
                "ovo_official_score": official_compatible_score(
                    task, response, ground_truth
                ),
                "ovo_strict_score": strict_score(task, response, ground_truth),
            }
        )
    return evaluated


def exclude_prefixed_videos(rows, prefixes):
    prefixes = tuple(prefix for prefix in prefixes if prefix)
    if not prefixes:
        return rows, 0
    kept = [
        row
        for row in rows
        if not str(row.get("video_id", "")).startswith(prefixes)
    ]
    return kept, len(rows) - len(kept)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(evaluated):
    task_official = defaultdict(list)
    task_strict = defaultdict(list)
    for row in evaluated:
        task = row["benchmark_task"]
        task_official[task].append(int(row["ovo_official_score"]))
        task_strict[task].append(int(row["ovo_strict_score"]))

    per_task = {}
    group_task_scores = defaultdict(list)
    group_task_strict_scores = defaultdict(list)
    for task in sorted(task_official):
        group = TASK_GROUPS[task]
        official_accuracy = mean(task_official[task])
        strict_accuracy = mean(task_strict[task])
        per_task[task] = {
            "group": group,
            "samples": len(task_official[task]),
            "official_accuracy": official_accuracy,
            "strict_accuracy": strict_accuracy,
        }
        group_task_scores[group].append(official_accuracy)
        group_task_strict_scores[group].append(strict_accuracy)

    per_group = {}
    for group in ("backward", "realtime", "forward"):
        if group not in group_task_scores:
            continue
        per_group[group] = {
            "tasks": len(group_task_scores[group]),
            "official_macro_accuracy": mean(group_task_scores[group]),
            "strict_macro_accuracy": mean(group_task_strict_scores[group]),
        }

    final_by_video = {}
    for row in evaluated:
        final_by_video[row.get("video_id", "")] = row
    input_tokens = sum(
        float(row.get("semantic_input_tokens", 0) or 0)
        for row in final_by_video.values()
    )
    written_tokens = sum(
        float(row.get("semantic_written_tokens", 0) or 0)
        for row in final_by_video.values()
    )
    total_patch_tokens = sum(
        int(float(row.get("vit_total_patch_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    updated_patch_tokens = sum(
        int(float(row.get("vit_updated_patch_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    output_input_tokens = sum(
        int(float(row.get("vit_output_input_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    output_tokens = sum(
        int(float(row.get("vit_output_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    kv_cache_memory_bytes = [
        int(float(row.get("kv_cache_memory_bytes", 0) or 0))
        for row in final_by_video.values()
    ]
    kv_cache_cpu_memory_bytes = [
        int(float(row.get("kv_cache_cpu_memory_bytes", 0) or 0))
        for row in final_by_video.values()
    ]
    kv_cache_gpu_memory_bytes = [
        int(float(row.get("kv_cache_gpu_memory_bytes", 0) or 0))
        for row in final_by_video.values()
    ]
    kv_cache_logical_tokens = [
        int(float(row.get("kv_cache_logical_tokens", 0) or 0))
        for row in final_by_video.values()
    ]
    semantic_timing_sec = {
        key: sum(
            float(row.get(f"semantic_{key}_sec", 0) or 0)
            for row in final_by_video.values()
        )
        for key in (
            "proposal",
            "preprocess",
            "embedding",
            "verification",
            "vit_encoder",
            "vision_backbone",
            "spatial_pool",
            "projector",
            "output_reduce",
            "context_write",
        )
    }
    visual_encoding_sec = sum(
        semantic_timing_sec[key]
        for key in ("preprocess", "embedding", "vit_encoder")
    )
    visual_selection_sec = sum(
        semantic_timing_sec[key]
        for key in ("proposal", "verification")
    )
    model_encoding_sec = sum(
        semantic_timing_sec[key]
        for key in ("embedding", "vit_encoder")
    )
    video_encode_wall_sec = sum(
        float(row.get("cumulative_encode_video_sec", 0) or 0)
        for row in final_by_video.values()
    )
    video_load_wall_sec = sum(
        float(row.get("load_video_sec", 0) or 0)
        for row in final_by_video.values()
    )
    init_prompt_wall_sec = sum(
        float(row.get("init_prompt_sec", 0) or 0)
        for row in final_by_video.values()
    )
    qa_wall_sec = sum(
        float(row.get("qa_sec", 0) or 0)
        for row in evaluated
    )
    full_pipeline_wall_sec = sum(
        float(row.get("elapsed_video_sec", 0) or 0)
        for row in final_by_video.values()
    )
    # 实时流研究不计离线视频文件加载；该指标只累计模型实际参与的在线处理阶段。
    online_model_pipeline_sec = (
        init_prompt_wall_sec + video_encode_wall_sec + qa_wall_sec
    )
    observed_stream_duration_sec = sum(
        (
            float(row.get("loaded_frames", 0) or 0)
            / float(row.get("sample_fps", 0) or 1)
        )
        for row in final_by_video.values()
    )
    semantic_input_frames = sum(
        int(float(row.get("semantic_input_frames", 0) or 0))
        for row in final_by_video.values()
    )
    arrived_frames = sum(
        int(float(row.get("loaded_frames", 0) or 0))
        for row in final_by_video.values()
    )
    profiled_stream_ingestion_sec = (
        visual_selection_sec
        + visual_encoding_sec
        + semantic_timing_sec["context_write"]
    )

    return {
        "samples": len(evaluated),
        "per_task": per_task,
        "per_group": per_group,
        "official_three_group_average": mean(
            [value["official_macro_accuracy"] for value in per_group.values()]
        ),
        "strict_three_group_average": mean(
            [value["strict_macro_accuracy"] for value in per_group.values()]
        ),
        # 保留旧字段兼容历史脚本；其准确含义是同步后的 encode_video 墙钟时间。
        "total_encode_video_sec": video_encode_wall_sec,
        "wall_clock_sec": {
            "online_video_processing": video_encode_wall_sec,
            "online_model_pipeline": online_model_pipeline_sec,
            "video_load": video_load_wall_sec,
            "init_prompt": init_prompt_wall_sec,
            "video_encode": video_encode_wall_sec,
            "qa": qa_wall_sec,
            "full_pipeline": full_pipeline_wall_sec,
            # 残差包含 Python 调度、张量索引、断言和未单独打点的运行时开销。
            "video_encode_unprofiled": max(
                0.0,
                video_encode_wall_sec - profiled_stream_ingestion_sec,
            ),
        },
        "semantic_input_frames": semantic_input_frames,
        "arrived_frames": arrived_frames,
        "semantic_kept_frames": sum(
            int(float(row.get("semantic_kept_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_candidate_frames": sum(
            int(float(row.get("semantic_candidate_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_preprocessed_frames": sum(
            int(float(row.get("semantic_preprocessed_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_token_reduction": (
            1.0 - written_tokens / input_tokens if input_tokens else 0.0
        ),
        "semantic_timing_sec": semantic_timing_sec,
        "latency_scope_sec": {
            # 视觉编码不包含帧选择；流式摄取包含选择、视觉编码和上下文写入。
            "visual_selection": visual_selection_sec,
            "model_encoding": model_encoding_sec,
            "visual_encoding": visual_encoding_sec,
            "stream_ingestion": profiled_stream_ingestion_sec,
        },
        "realtime_metrics": {
            "observed_stream_duration_sec": observed_stream_duration_sec,
            "online_processing_fps": (
                arrived_frames / video_encode_wall_sec
                if video_encode_wall_sec
                else 0.0
            ),
            "realtime_compute_ratio": (
                video_encode_wall_sec / observed_stream_duration_sec
                if observed_stream_duration_sec
                else 0.0
            ),
        },
        "metric_definitions": {
            "official_three_group_average": (
                "OVO-Bench 三个任务组的宏平均准确率；先对每个任务求准确率，"
                "再在组内和三个组之间等权平均。"
            ),
            "model_encoding": (
                "模型内部视觉编码时间：patch embedding（补丁嵌入）"
                "加 ViT encoder（视觉编码器）；不含图像预处理和上下文写入。"
            ),
            "visual_encoding": (
                "视觉编码时间：图像预处理加 model_encoding；"
                "不含帧选择和语言模型上下文写入。"
            ),
            "stream_ingestion": (
                "已打点的流式摄入时间：帧选择、视觉编码和视觉 token 上下文写入之和。"
            ),
            "online_video_processing": (
                "论文主效率指标：实时帧到达系统后，从调用 encode_video 到完成视觉编码、"
                "语义筛选和上下文写入的同步墙钟时间；不含视频文件读取、帧到达等待、"
                "初始化提示和 QA。"
            ),
            "online_model_pipeline": (
                "在线模型总处理时间：初始化提示、online_video_processing 和 QA 之和；"
                "不含离线视频文件读取与真实时间轴上的帧到达等待。"
            ),
            "observed_stream_duration_sec": (
                "输入帧按采样 FPS 对应的真实时间轴长度；用于判断计算能否跟上实时输入，"
                "不是程序等待时间。"
            ),
            "online_processing_fps": (
                "实际到达帧数除以 online_video_processing；表示模型在线摄入吞吐，"
                "不依赖是否启用语义门控。"
            ),
            "realtime_compute_ratio": (
                "online_video_processing 除以输入流时长；小于 1 表示计算速度能跟上实时流。"
            ),
            "video_encode": (
                "online_video_processing 的兼容别名。"
            ),
            "full_pipeline": (
                "离线数据适配器诊断时间：每个视频从文件读取到最后一个 QA 完成；"
                "包含与实时流系统无关的视频文件读取，禁止用于论文主加速比。"
            ),
            "context_write": (
                "将保留的视觉 token 送入语言模型并更新 ReKV/KV cache 的同步时间。"
            ),
            "kv_cache_memory": (
                "QA 解码前测得的 ReKV/KV cache 实际字节数；"
                "均值用于总体比较，峰值用于长视频容量分析。"
            ),
        },
        "paper_reporting_policy": {
            "system_input_contract": (
                "输入是实时采集链路已经交付、已解码并按时间顺序到达的 RGB 帧；"
                "论文方法从帧到达模型侧开始计时。"
            ),
            "primary_latency_metric": "wall_clock_sec.online_video_processing",
            "system_latency_metric": "wall_clock_sec.online_model_pipeline",
            "supporting_breakdown": "latency_scope_sec.stream_ingestion",
            "realtime_capacity_metric": "realtime_metrics.realtime_compute_ratio",
            # 离线读取和解码只反映 benchmark 适配器，不代表实时流模型系统本身。
            "excluded_from_speedup": [
                "wall_clock_sec.video_load",
                "wall_clock_sec.full_pipeline",
                "camera_capture",
                "network_transport",
                "video_decode",
                "frame_arrival_wait",
            ],
        },
        "vit_layer_sparse": {
            "dense_frames": sum(
                int(float(row.get("vit_dense_frames", 0) or 0))
                for row in final_by_video.values()
            ),
            "sparse_frames": sum(
                int(float(row.get("vit_sparse_frames", 0) or 0))
                for row in final_by_video.values()
            ),
            "dense_sec": sum(
                float(row.get("vit_dense_sec", 0) or 0)
                for row in final_by_video.values()
            ),
            "sparse_sec": sum(
                float(row.get("vit_sparse_sec", 0) or 0)
                for row in final_by_video.values()
            ),
            "total_patch_tokens": total_patch_tokens,
            "updated_patch_tokens": updated_patch_tokens,
            "planned_update_ratio": (
                updated_patch_tokens / total_patch_tokens
                if total_patch_tokens
                else 0.0
            ),
        },
        "vit_output_reduction": {
            "policy": next(
                (
                    row.get("vit_output_policy", "")
                    for row in final_by_video.values()
                    if row.get("vit_output_policy", "")
                ),
                "",
            ),
            "budget_per_frame": max(
                (
                    int(float(row.get("vit_output_budget_per_frame", 0) or 0))
                    for row in final_by_video.values()
                ),
                default=0,
            ),
            "base_tokens_per_frame": max(
                (
                    int(
                        float(
                            row.get(
                                "vit_output_base_tokens_per_frame",
                                0,
                            )
                            or 0
                        )
                    )
                    for row in final_by_video.values()
                ),
                default=0,
            ),
            "residual_tokens_per_frame": max(
                (
                    int(
                        float(
                            row.get(
                                "vit_output_residual_tokens_per_frame",
                                0,
                            )
                            or 0
                        )
                    )
                    for row in final_by_video.values()
                ),
                default=0,
            ),
            "input_tokens": output_input_tokens,
            "output_tokens": output_tokens,
            "reduction_ratio": (
                1.0 - output_tokens / output_input_tokens
                if output_input_tokens
                else 0.0
            ),
        },
        "kv_cache_memory": {
            # 均值用于跨方法比较，峰值用于检查长视频的缓存压力。
            "mean_bytes": (
                mean(kv_cache_memory_bytes)
                if kv_cache_memory_bytes
                else 0.0
            ),
            "max_bytes": max(kv_cache_memory_bytes, default=0),
            "mean_cpu_bytes": mean(kv_cache_cpu_memory_bytes),
            "max_cpu_bytes": max(kv_cache_cpu_memory_bytes, default=0),
            "mean_gpu_bytes": mean(kv_cache_gpu_memory_bytes),
            "max_gpu_bytes": max(kv_cache_gpu_memory_bytes, default=0),
            "mean_logical_tokens": mean(kv_cache_logical_tokens),
            "max_logical_tokens": max(kv_cache_logical_tokens, default=0),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ReKV CSV predictions with the OVO-Bench offline protocol."
    )
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--exclude-video-prefix",
        action="append",
        default=["warmup-"],
        help="汇总前排除指定 video_id 前缀；默认排除预热样本。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, excluded_rows = exclude_prefixed_videos(
        read_csv(args.pred_path),
        args.exclude_video_prefix,
    )
    evaluated = evaluate_rows(rows)
    summary = summarize(evaluated)
    summary["pred_path"] = args.pred_path
    summary["excluded_rows"] = excluded_rows
    write_csv(args.output_csv, evaluated)
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
