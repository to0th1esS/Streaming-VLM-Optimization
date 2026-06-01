param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-gpu",
    [string]$OutputDir = "results\turbovit_v1\v4_semantic",
    [string]$Backbone = "clip",
    [string]$ModelPath = "/home/mllm/models/clip-vit-large-patch14-336",
    [string]$VideoSource = "real",
    [string]$VideoPath = "data\turbovit_v1\big_buck_bunny.mp4",
    [int]$NumFrames = 48,
    [int]$FrameStride = 1,
    [int]$RefreshInterval = 4,
    [double]$SparseRatioMin = 0.75,
    [double]$SparseRatioMax = 1.0,
    [int]$ProbeLayer = 2,
    [double]$SkipPatchThreshold = 0.001,
    [double]$DensePatchThreshold = 0.006,
    [double]$SkipFeatureThreshold = 0.9995,
    [double]$DenseFeatureThreshold = 0.98,
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

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_turbovit_v4 `
    --output-dir $OutputDir `
    --backbone $Backbone `
    --model-path $ModelPath `
    --video-source $VideoSource `
    --video-path $VideoPath `
    --num-frames $NumFrames `
    --frame-stride $FrameStride `
    --refresh-interval $RefreshInterval `
    --sparse-ratio-min $SparseRatioMin `
    --sparse-ratio-max $SparseRatioMax `
    --probe-layer $ProbeLayer `
    --skip-patch-threshold $SkipPatchThreshold `
    --dense-patch-threshold $DensePatchThreshold `
    --skip-feature-threshold $SkipFeatureThreshold `
    --dense-feature-threshold $DenseFeatureThreshold `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Turbo-ViT-v4 local run failed with exit code $LASTEXITCODE"
}
