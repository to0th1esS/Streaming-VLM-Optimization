param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-gpu",
    [string]$OutputDir = "results\turbovit_v1\synthetic_vit_v2_gpu",
    [string]$Weights = "none",
    [int]$NumFrames = 24,
    [int]$RefreshInterval = 4,
    [double]$DynamicRatio = 0.75,
    [string]$SkipThresholds = "0.0001,0.0005,0.001",
    [double]$DenseThreshold = 0.006,
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

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_synthetic_vit_v2 `
    --output-dir $OutputDir `
    --weights $Weights `
    --num-frames $NumFrames `
    --refresh-interval $RefreshInterval `
    --dynamic-ratio $DynamicRatio `
    --skip-thresholds $SkipThresholds `
    --dense-threshold $DenseThreshold `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Synthetic ViT v2 local run failed with exit code $LASTEXITCODE"
}
