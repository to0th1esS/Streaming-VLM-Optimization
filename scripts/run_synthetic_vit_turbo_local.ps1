param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-gpu",
    [string]$OutputDir = "results\turbovit_v1\synthetic_vit_turbo_gpu",
    [string]$Weights = "none",
    [int]$NumFrames = 24,
    [int]$RefreshInterval = 4,
    [string]$DynamicRatios = "0.25,0.5,0.75",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$MicromambaExe = Join-Path $Root ".tools\Library\bin\micromamba.exe"
$EnvPath = Join-Path $Root $EnvPrefix

if (!(Test-Path $MicromambaExe)) {
    throw "Missing local micromamba. Run .\run_certification_experiment.ps1 first."
}

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_synthetic_vit_turbo `
    --output-dir $OutputDir `
    --weights $Weights `
    --num-frames $NumFrames `
    --refresh-interval $RefreshInterval `
    --dynamic-ratios $DynamicRatios `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Synthetic ViT Turbo local run failed with exit code $LASTEXITCODE"
}
