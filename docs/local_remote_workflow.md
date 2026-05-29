# Local-to-remote experiment workflow

This repository is split into two roles:

- Local machine: fast code iteration and small certification experiments.
- Remote GPU server: large-scale evaluation with real model weights and datasets.

Only code, lightweight scripts, and configuration templates should move through
git. Conda environments, downloaded tools, model weights, datasets, logs, and
large results stay local to each machine.

## Local loop

Run the lightweight certification before committing:

```powershell
.\run_certification_experiment.ps1
```

For the fixed local-to-remote pipeline, run:

```powershell
.\run_research_pipeline.ps1
```

This performs local certification, commits changed code, pushes to the `server`
remote, prepares remote model links under `/home/mllm/models`, and runs remote
certification.

To also launch a remote large-scale evaluation after certification:

```powershell
.\run_research_pipeline.ps1 -RunRemoteEval `
  -RemoteEvalCondaEnv rekv `
  -Model llava_ov_0.5b `
  -Dataset qaego4d
```

The script uses:

```text
.tools/                         local micromamba, ignored by git
.conda-envs/vit-sparse-cert     local conda env, ignored by git
results/vit_sparse_certification.json
```

To reuse the same runner for a future local certification script:

```powershell
.\run_certification_experiment.ps1 `
  -VerificationScript scripts\verify_my_new_method.py `
  -ExperimentName my_new_method_cert `
  -ResultPath results\my_new_method_cert.json
```

## Commit boundary

Commit only source changes and lightweight reproducibility files:

```text
model/
video_qa/
scripts/
docs/
environment*.yml
run_certification_experiment.*
```

Do not commit:

```text
.tools/
.conda-envs/
data/
model_zoo/
results/
*.log
*.json
*.jsonl
```

## Remote loop

On the remote server, keep a full GPU environment and real assets outside git:

```text
model_zoo/   server-only model weights
data/        server-only datasets
results/     server-only large outputs
```

Use the remote template:

```bash
REPO_DIR=/path/to/Streaming-VLM-Optimization \
CONDA_BIN=/root/miniconda3/bin/conda \
CONDA_ENV=rekv \
MODEL=llava_ov_0.5b \
DATASET=qaego4d \
bash scripts/run_remote_eval_template.sh
```

The script does a fast-forward `git pull` and then runs `video_qa.run_eval` in
the server's own conda environment.

For a remote lightweight certification without model weights:

```bash
REPO_DIR=/path/to/Streaming-VLM-Optimization \
CONDA_BIN=/root/miniconda3/bin/conda \
CONDA_ENV=base \
bash scripts/run_remote_certification.sh
```

Prepare model entries under `/home/mllm/models`:

```bash
REPO_DIR=/home/yangjin/1#Streaming-VLM-Optimization
cd "$REPO_DIR"
MODEL_ROOT=/home/mllm/models \
MIRROR_ROOT=/home/Streaming-VLM-Optimization/model_zoo \
REPO_MODEL_ZOO=model_zoo \
bash scripts/setup_remote_models.sh
```

By default this prepares:

```text
llava-onevision-qwen2-0.5b-ov-hf
llava-onevision-qwen2-7b-ov-hf
Video-LLaVA-7B-hf
LongVA-7B
```

If a model already exists in `/home/Streaming-VLM-Optimization/model_zoo`, the
script links it into `/home/mllm/models` instead of duplicating the checkpoint.
It also links the current repository's `model_zoo/<name>` to `/home/mllm/models/<name>`.

To force a physical Hugging Face download into `/home/mllm/models`, bypassing
existing mirrors:

```bash
FORCE_DOWNLOAD=1 \
MODEL_ROOT=/home/mllm/models \
bash scripts/setup_remote_models.sh
```

## Recommended iteration

1. Modify code locally.
2. Run `.\run_certification_experiment.ps1`.
3. Commit and push if the local certification passes.
4. On the remote server, run `scripts/run_remote_eval_template.sh`.
5. Keep large outputs on the server; summarize only useful metrics back into
   notes or future lightweight config files.
