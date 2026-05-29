param(
    [string]$EnvName = "streaming-vlm-opt"
)

$ErrorActionPreference = "Stop"

conda run -n $EnvName python scripts/verify_vit_sparse_patch.py
