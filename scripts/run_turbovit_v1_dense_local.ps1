param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-cert",
    [string]$OutputDir = "results\turbovit_v1\v0_dense_baseline",
    [int]$NumFrames = 24,
    [int]$Depth = 6,
    [int]$EmbedDim = 96
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$MicromambaExe = Join-Path $Root ".tools\Library\bin\micromamba.exe"
$EnvPath = Join-Path $Root $EnvPrefix

if (!(Test-Path $MicromambaExe)) {
    throw "Missing local micromamba. Run .\run_certification_experiment.ps1 first."
}

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_dense_baseline `
    --num-frames $NumFrames `
    --depth $Depth `
    --embed-dim $EmbedDim `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Turbo-ViT-v1 dense local run failed with exit code $LASTEXITCODE"
}
