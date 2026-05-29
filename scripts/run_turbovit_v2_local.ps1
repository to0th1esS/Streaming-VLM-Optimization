param(
    [string]$EnvPrefix = ".conda-envs\vit-sparse-cert",
    [string]$OutputDir = "results\turbovit_v1\v2_segment_decision",
    [string]$VideoSource = "real",
    [int]$RefreshInterval = 4,
    [double]$DynamicRatio = 0.75,
    [double]$SkipThreshold = 0.0005,
    [double]$DenseThreshold = 0.006
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$MicromambaExe = Join-Path $Root ".tools\Library\bin\micromamba.exe"
$EnvPath = Join-Path $Root $EnvPrefix

if (!(Test-Path $MicromambaExe)) {
    throw "Missing local micromamba. Run .\run_certification_experiment.ps1 first."
}

& $MicromambaExe run -p $EnvPath python -m experiments.turbovit_v1.scripts.run_turbovit_v2 `
    --video-source $VideoSource `
    --refresh-interval $RefreshInterval `
    --dynamic-ratio $DynamicRatio `
    --skip-threshold $SkipThreshold `
    --dense-threshold $DenseThreshold `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Turbo-ViT-v2 local run failed with exit code $LASTEXITCODE"
}
