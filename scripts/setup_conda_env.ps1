param(
    [string]$EnvName = "streaming-vlm-opt"
)

$ErrorActionPreference = "Stop"

conda env create -n $EnvName -f environment.yml
conda run -n $EnvName python scripts/verify_vit_sparse_patch.py

Write-Host "Conda environment '$EnvName' is ready."
