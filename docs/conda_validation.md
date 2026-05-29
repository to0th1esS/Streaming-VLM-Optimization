# Conda validation environment

This environment is intended to validate the local ViT sparse-update patch without
downloading model checkpoints or datasets.

## One-command run

From the repository root:

```powershell
.\run_certification_experiment.ps1
```

On Windows Explorer, you can also double-click:

```text
run_certification_experiment.bat
```

The script downloads a local micromamba binary into `.tools/`, creates the conda
environment under `.conda-envs/vit-sparse-cert`, runs the certification
experiment, and writes:

```text
results/vit_sparse_certification.json
```

## Manual environment creation

```powershell
conda env create -n streaming-vlm-opt -f environment.yml
```

If the environment already exists:

```powershell
conda env update -n streaming-vlm-opt -f environment.yml --prune
```

## Run the local verification

```powershell
conda run -n streaming-vlm-opt python scripts/verify_vit_sparse_patch.py
```

Expected output:

```text
vit sparse patch verification passed
```

## What this verifies

- `vit_patch_hf` can patch a model-like object.
- ViT layer dense/reference update and sparse update both execute.
- `encode_video` updates the inference context per chunk.
- `vit_output_postprocess` is called and can change the ViT token sequence before
  the final reshape.

## What this does not verify

- Real LLaVA-OneVision checkpoint loading.
- Dataset decoding and benchmark evaluation.
- End-to-end GPU memory behavior.
- Optional `flash-attn` or Triton kernels.

For full benchmark evaluation, place model weights under `model_zoo/`, datasets
under `data/`, and run `python -m video_qa.run_eval ...` from the conda
environment. The original project targets Linux with CUDA GPUs; Windows local
validation should start with the lightweight script above.
