param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-cert",
    [string]$OutputDir = "results\turbovit_v1\real_vit_turbo",
    [string]$Weights = "none",
    [int]$NumFrames = 6,
    [int]$FrameStride = 4,
    [int]$RefreshInterval = 4,
    [double]$DynamicRatio = 0.5,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$MicromambaExe = Join-Path $Root ".tools\Library\bin\micromamba.exe"
$EnvPath = Join-Path $Root $EnvPrefix

if (!(Test-Path $MicromambaExe)) {
    throw "Missing local micromamba. Run .\run_certification_experiment.ps1 first."
}

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_real_vit_turbo `
    --output-dir $OutputDir `
    --weights $Weights `
    --num-frames $NumFrames `
    --frame-stride $FrameStride `
    --refresh-interval $RefreshInterval `
    --dynamic-ratio $DynamicRatio `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Real ViT Turbo local run failed with exit code $LASTEXITCODE"
}
