import os
import argparse
import subprocess
import multiprocessing


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)


def append_semantic_stream_args(cmd, args):
    cmd.extend([
        "--enable_vit_sparse", str(args.enable_vit_sparse),
        "--enable_vit_layer_sparse", str(args.enable_vit_layer_sparse),
        "--vit_cache_interval", str(args.vit_cache_interval),
        "--vit_update_token_ratio", str(args.vit_update_token_ratio),
        "--vit_output_token_policy", str(args.vit_output_token_policy),
        "--vit_output_token_budget", str(args.vit_output_token_budget),
        "--vit_output_base_tokens", str(args.vit_output_base_tokens),
        "--vit_output_coverage_tokens", str(args.vit_output_coverage_tokens),
        "--vit_output_drift_dims", str(args.vit_output_drift_dims),
        "--vit_output_selection_space", str(args.vit_output_selection_space),
        "--enable_semantic_stream", str(args.enable_semantic_stream),
        "--enable_semantic_compute_gate", str(args.enable_semantic_compute_gate),
        "--semantic_refresh_interval", str(args.semantic_refresh_interval),
        "--semantic_skip_threshold", str(args.semantic_skip_threshold),
        "--semantic_recency_keep_frames", str(args.semantic_recency_keep_frames),
        "--semantic_recency_updates_anchor", str(args.semantic_recency_updates_anchor),
        "--semantic_coverage_interval", str(args.semantic_coverage_interval),
        "--semantic_coverage_updates_anchor", str(args.semantic_coverage_updates_anchor),
        "--semantic_selection_policy", str(args.semantic_selection_policy),
        "--semantic_selection_feature_source", str(args.semantic_selection_feature_source),
        "--semantic_candidate_multiplier", str(args.semantic_candidate_multiplier),
        "--semantic_raw_signature_mode", str(args.semantic_raw_signature_mode),
        "--semantic_raw_grid_size", str(args.semantic_raw_grid_size),
        "--semantic_raw_proposal_policy", str(args.semantic_raw_proposal_policy),
        "--semantic_saliency_z_threshold", str(args.semantic_saliency_z_threshold),
        "--semantic_pair_similarity_threshold", str(args.semantic_pair_similarity_threshold),
        "--semantic_profile_breakdown", str(args.semantic_profile_breakdown),
        "--semantic_budget_window_size", str(args.semantic_budget_window_size),
        "--semantic_budget_keep_per_window", str(args.semantic_budget_keep_per_window),
        "--enable_query_aware_retrieval", str(args.enable_query_aware_retrieval),
        "--query_retrieval_policy", str(args.query_retrieval_policy),
        "--latest_retrieval_blocks", str(args.latest_retrieval_blocks),
        "--latest_query_terms", str(args.latest_query_terms),
        "--qa_max_new_tokens", str(args.qa_max_new_tokens),
    ])
    return cmd


def eval_mlvu(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/mlvu/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/mlvu/dev_debug_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            cmd = append_semantic_stream_args(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/qaego4d/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/qaego4d/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            cmd = append_semantic_stream_args(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_egoschema(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/egoschema/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/egoschema/full.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_activitynet_qa(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/activitynet_qa/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_ego(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/rvs_ego/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            cmd = append_semantic_stream_args(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_movie(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/rvs_movie/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            cmd = append_semantic_stream_args(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_cgbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/cgbench/{args.retrieve_size}-{args.sample_fps}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/cgbench/full_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--qa_max_new_tokens", type=int, default=256)
    parser.add_argument("--enable_vit_sparse", type=str, default='true')
    parser.add_argument("--enable_vit_layer_sparse", type=str, default='true')
    parser.add_argument("--vit_cache_interval", type=int, default=2)
    parser.add_argument("--vit_update_token_ratio", type=float, default=0.25)
    parser.add_argument(
        "--vit_output_token_policy",
        type=str,
        default="none",
        choices=[
            "none",
            "uniform",
            "coverage_innovation",
            "structured_pool",
            "structured_residual",
            "post_projector_pool",
            "post_projector_sample",
        ],
    )
    parser.add_argument("--vit_output_token_budget", type=int, default=196)
    parser.add_argument("--vit_output_base_tokens", type=int, default=100)
    parser.add_argument("--vit_output_coverage_tokens", type=int, default=16)
    parser.add_argument("--vit_output_drift_dims", type=int, default=0)
    parser.add_argument(
        "--vit_output_selection_space",
        type=str,
        default="projected",
        choices=["projected", "vit_native"],
    )
    parser.add_argument("--enable_semantic_stream", type=str, default='false')
    parser.add_argument("--enable_semantic_compute_gate", type=str, default='false')
    parser.add_argument("--semantic_refresh_interval", type=int, default=4)
    parser.add_argument("--semantic_skip_threshold", type=float, default=0.01)
    parser.add_argument("--semantic_recency_keep_frames", type=int, default=0)
    parser.add_argument("--semantic_recency_updates_anchor", type=str, default='false')
    parser.add_argument("--semantic_coverage_interval", type=int, default=0)
    parser.add_argument("--semantic_coverage_updates_anchor", type=str, default='false')
    parser.add_argument("--semantic_selection_policy", type=str, default='threshold', choices=['threshold', 'budget_topk', 'periodic'])
    parser.add_argument("--semantic_selection_feature_source", type=str, default='vit_embedding', choices=['vit_embedding', 'raw_rgb', 'hybrid'])
    parser.add_argument("--semantic_candidate_multiplier", type=int, default=4)
    parser.add_argument("--semantic_raw_signature_mode", type=str, default='avg_pool', choices=['avg_pool', 'grid_sample', 'grid_sample_stable'])
    parser.add_argument("--semantic_raw_grid_size", type=int, default=4)
    parser.add_argument("--semantic_raw_proposal_policy", type=str, default='novelty_topk', choices=['novelty_topk', 'saliency_gated', 'saliency_paired'])
    parser.add_argument("--semantic_saliency_z_threshold", type=float, default=4.0)
    parser.add_argument("--semantic_pair_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--semantic_profile_breakdown", type=str, default='false')
    parser.add_argument("--semantic_budget_window_size", type=int, default=0)
    parser.add_argument("--semantic_budget_keep_per_window", type=int, default=1)
    parser.add_argument("--enable_query_aware_retrieval", type=str, default='false')
    parser.add_argument(
        "--query_retrieval_policy",
        choices=["internal", "latest_recent", "always_recent"],
        default="latest_recent",
    )
    parser.add_argument("--latest_retrieval_blocks", type=int, default=0)
    parser.add_argument(
        "--latest_query_terms",
        type=str,
        default="latest,current,currently,now,setting,where,last frame,latest clip,latest video frame",
    )
    parser.add_argument("--debug", type=str, default='false')
    args = parser.parse_args()
    func_dic = {
        'mlvu': eval_mlvu,
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
