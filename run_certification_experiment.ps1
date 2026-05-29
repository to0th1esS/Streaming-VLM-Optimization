param(
    [string]$EnvName = "vit-sparse-cert",
    [string]$EnvPrefix = ".conda-envs\vit-sparse-cert",
    [string]$VerificationScript = "scripts\verify_vit_sparse_patch.py",
    [string]$ExperimentName = "vit_sparse_certification",
    [string]$ResultPath = "results\vit_sparse_certification.json"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ToolsDir = Join-Path $Root ".tools"
$MicromambaExe = Join-Path $ToolsDir "Library\bin\micromamba.exe"
$EnvPath = Join-Path $Root $EnvPrefix

if (!(Test-Path $MicromambaExe)) {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $Archive = Join-Path $ToolsDir "micromamba.tar.bz2"
    Write-Host "Downloading micromamba..."
    Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile $Archive
    tar -xjf $Archive -C $ToolsDir
}

if (!(Test-Path $EnvPath)) {
    Write-Host "Creating conda environment at $EnvPath ..."
    & $MicromambaExe create -y -p $EnvPath -f environment.cert.yml
    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment creation failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "Using existing conda environment at $EnvPath"
}

& $MicromambaExe run -p $EnvPath python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing torch package..."
    & $MicromambaExe run -p $EnvPath python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Torch installation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Running certification experiment..."
& $MicromambaExe run -p $EnvPath python $VerificationScript --experiment-name $ExperimentName --result-path $ResultPath
if ($LASTEXITCODE -ne 0) {
    throw "Certification experiment failed with exit code $LASTEXITCODE"
}

Write-Host "Certification result:"
Get-Content $ResultPath
