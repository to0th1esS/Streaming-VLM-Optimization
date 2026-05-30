param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-gpu",
    [string]$OutputDir = "results\turbovit_v1\v3_analysis",
    [string]$Backbone = "clip",
    [string]$ModelPath = "/home/mllm/models/clip-vit-large-patch14-336",
    [string]$VideoSource = "real",
    [string]$VideoPath = "data\turbovit_v1\big_buck_bunny.mp4",
    [int]$NumFrames = 48,
    [int]$FrameStride = 4,
    [int]$RefreshInterval = 4,
    [double]$DynamicRatio = 0.75,
    [double]$SkipThreshold = 0.001,
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

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_v3_analysis `
    --output-dir $OutputDir `
    --backbone $Backbone `
    --model-path $ModelPath `
    --video-source $VideoSource `
    --video-path $VideoPath `
    --num-frames $NumFrames `
    --frame-stride $FrameStride `
    --refresh-interval $RefreshInterval `
    --dynamic-ratio $DynamicRatio `
    --skip-threshold $SkipThreshold `
    --dense-threshold $DenseThreshold `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Turbo-ViT v3 analysis local run failed with exit code $LASTEXITCODE"
}
