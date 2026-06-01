import argparse
import csv
from pathlib import Path

from experiments.turbovit_v1.eval.cache_policy_sim import (
    simulate_visual_cache_policy,
    summarize_many,
    write_cache_policy_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate visual-token cache reuse from Turbo-ViT traces.")
    parser.add_argument("result_dirs", nargs="+", help="Turbo-ViT result directories containing v*_latency.csv.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/cache_policy_sim")
    parser.add_argument("--visual-tokens-per-frame", type=int, default=576)
    parser.add_argument("--cache-reuse-overhead-ratio", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = [Path(item) for item in args.result_dirs]
    for result_dir in result_dirs:
        simulation = simulate_visual_cache_policy(
            result_dir,
            visual_tokens_per_frame=args.visual_tokens_per_frame,
            cache_reuse_overhead_ratio=args.cache_reuse_overhead_ratio,
        )
        write_cache_policy_outputs(output_dir / result_dir.name, simulation)

    rows = summarize_many(result_dirs, visual_tokens_per_frame=args.visual_tokens_per_frame)
    if rows:
        with (output_dir / "cache_policy_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Cache policy simulation completed")
    print(f"summary: {output_dir / 'cache_policy_comparison.csv'}")
    for row in rows:
        print(
            f"{Path(row['result_dir']).name}: "
            f"token reduction={row['raw_visual_token_reduction']:.3f}, "
            f"vit speedup={row['vit_speedup']:.3f}, "
            f"cos={row['mean_output_cosine']:.6f}"
        )


if __name__ == "__main__":
    main()
