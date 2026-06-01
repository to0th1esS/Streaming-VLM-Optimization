import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def _find_latency_csv(result_dir: Path) -> Path:
    for name in ("v7_latency.csv", "v6_latency.csv", "v5_latency.csv", "v4_latency.csv"):
        path = result_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no Turbo-ViT latency csv found in {result_dir}")


def _find_summary_json(result_dir: Path) -> Path:
    for name in ("v7_summary.json", "v6_summary.json", "v5_summary.json", "v4_summary.json"):
        path = result_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no Turbo-ViT summary json found in {result_dir}")


def simulate_visual_cache_policy(
    result_dir: Path,
    visual_tokens_per_frame: int = 576,
    cache_reuse_overhead_ratio: float = 0.0,
) -> Dict:
    latency_rows = _read_csv(_find_latency_csv(result_dir))
    summary = _read_json(_find_summary_json(result_dir))

    dense_tokens = 0.0
    rewritten_tokens = 0.0
    reused_tokens = 0.0
    decision_counts: Dict[str, int] = {}
    weighted_dynamic_ratio = 0.0
    sparse_frames = 0

    frame_rows = []
    for row in latency_rows:
        decision = row["decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        dense_tokens += visual_tokens_per_frame

        if decision == "dense":
            frame_rewritten = float(visual_tokens_per_frame)
        elif decision == "skip":
            frame_rewritten = 0.0
        else:
            dynamic_ratio = float(row.get("dynamic_ratio_observed", 1.0))
            frame_rewritten = visual_tokens_per_frame * min(1.0, max(0.0, dynamic_ratio))
            weighted_dynamic_ratio += dynamic_ratio
            sparse_frames += 1

        frame_reused = visual_tokens_per_frame - frame_rewritten
        rewritten_tokens += frame_rewritten
        reused_tokens += frame_reused
        frame_rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "decision": decision,
                "dense_visual_tokens": visual_tokens_per_frame,
                "rewritten_visual_tokens": frame_rewritten,
                "reused_visual_tokens": frame_reused,
                "dynamic_ratio_observed": float(row.get("dynamic_ratio_observed", 1.0)),
                "semantic_stability": float(row.get("semantic_stability", 1.0)),
            }
        )

    effective_rewrite_tokens = rewritten_tokens + reused_tokens * cache_reuse_overhead_ratio
    reduction = 1.0 - effective_rewrite_tokens / dense_tokens if dense_tokens else 0.0
    raw_reduction = 1.0 - rewritten_tokens / dense_tokens if dense_tokens else 0.0

    return {
        "source_result_dir": str(result_dir),
        "experiment": summary.get("experiment", ""),
        "num_frames": len(latency_rows),
        "visual_tokens_per_frame": visual_tokens_per_frame,
        "decision_counts": decision_counts,
        "dense_visual_tokens": dense_tokens,
        "rewritten_visual_tokens": rewritten_tokens,
        "reused_visual_tokens": reused_tokens,
        "cache_reuse_overhead_ratio": cache_reuse_overhead_ratio,
        "effective_rewrite_tokens": effective_rewrite_tokens,
        "raw_visual_token_reduction": raw_reduction,
        "effective_visual_token_reduction": reduction,
        "mean_sparse_dynamic_ratio": weighted_dynamic_ratio / sparse_frames if sparse_frames else 0.0,
        "vit_speedup": summary.get("speedup", 0.0),
        "mean_output_cosine": summary.get("mean_output_cosine", 0.0),
        "false_skip_count": summary.get("false_skip_count", 0),
        "frame_rows": frame_rows,
    }


def write_cache_policy_outputs(output_dir: Path, simulation: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = simulation.pop("frame_rows")
    with (output_dir / "cache_policy_summary.json").open("w") as handle:
        json.dump(simulation, handle, indent=2)
    if frame_rows:
        with (output_dir / "cache_policy_frames.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    simulation["frame_rows"] = frame_rows


def summarize_many(result_dirs: Iterable[Path], visual_tokens_per_frame: int = 576) -> List[Dict]:
    rows = []
    for result_dir in result_dirs:
        simulation = simulate_visual_cache_policy(result_dir, visual_tokens_per_frame)
        rows.append(
            {
                "result_dir": str(result_dir),
                "experiment": simulation["experiment"],
                "num_frames": simulation["num_frames"],
                "decision_counts": json.dumps(simulation["decision_counts"], sort_keys=True),
                "vit_speedup": simulation["vit_speedup"],
                "mean_output_cosine": simulation["mean_output_cosine"],
                "raw_visual_token_reduction": simulation["raw_visual_token_reduction"],
                "mean_sparse_dynamic_ratio": simulation["mean_sparse_dynamic_ratio"],
            }
        )
    return rows
