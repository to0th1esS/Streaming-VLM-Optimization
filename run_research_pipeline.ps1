param(
    [string]$RemoteName = "server",
    [string]$RemoteHost = "remote-docker",
    [string]$RemoteRepoDir = "/home/yangjin/1#Streaming-VLM-Optimization",
    [string]$RemoteCondaBin = "/root/miniconda3/bin/conda",
    [string]$RemoteCondaEnv = "base",
    [string]$CommitMessage = "",
    [switch]$SkipCommit,
    [switch]$SkipModelSetup,
    [switch]$RunRemoteEval,
    [string]$RemoteEvalCondaEnv = "rekv",
    [string]$Model = "llava_ov_0.5b",
    [string]$Dataset = "qaego4d",
    [string]$NumChunks = "1",
    [string]$SampleFps = "0.5",
    [string]$NLocal = "15000",
    [string]$RetrieveSize = "64"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Run-Checked {
    param([string]$Command)
    Write-Host "`n> $Command"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command"
    }
}

Write-Host "Step 1/5: local certification"
Run-Checked "powershell -NoProfile -ExecutionPolicy Bypass -File .\run_certification_experiment.ps1"

if (-not $SkipCommit) {
    Write-Host "`nStep 2/5: commit local code if needed"
    $Status = git -c safe.directory=$Root status --short
    if ($Status) {
        git -c safe.directory=$Root add -A
        if (-not $CommitMessage) {
            $CommitMessage = "Research sync $(Get-Date -Format 'yyyyMMdd-HHmmss')"
        }
        Run-Checked "git -c safe.directory=`"$Root`" commit -m `"$CommitMessage`""
    } else {
        Write-Host "No local changes to commit."
    }
} else {
    Write-Host "`nStep 2/5: commit skipped"
}

Write-Host "`nStep 3/5: push to $RemoteName"
$Branch = git -c safe.directory=$Root branch --show-current
Run-Checked "git -c safe.directory=`"$Root`" push $RemoteName $Branch"

if (-not $SkipModelSetup) {
    Write-Host "`nStep 4/5: remote model setup under /home/models"
    $ModelSetupCmd = "cd '$RemoteRepoDir' && MODEL_ROOT=/home/models MIRROR_ROOT=/home/Streaming-VLM-Optimization/model_zoo bash scripts/setup_remote_models.sh"
    Run-Checked "ssh -o BatchMode=yes $RemoteHost `"$ModelSetupCmd`""
} else {
    Write-Host "`nStep 4/5: remote model setup skipped"
}

Write-Host "`nStep 5/5: remote certification"
$RemoteCertCmd = "cd '$RemoteRepoDir' && REPO_DIR='$RemoteRepoDir' CONDA_BIN=$RemoteCondaBin CONDA_ENV=$RemoteCondaEnv bash scripts/run_remote_certification.sh"
Run-Checked "ssh -o BatchMode=yes $RemoteHost `"$RemoteCertCmd`""

if ($RunRemoteEval) {
    Write-Host "`nStep 6/6: remote large-scale eval with pre-certification"
    $RemoteEvalCmd = "cd '$RemoteRepoDir' && REPO_DIR='$RemoteRepoDir' CONDA_BIN=$RemoteCondaBin CONDA_ENV=$RemoteEvalCondaEnv MODEL=$Model DATASET=$Dataset NUM_CHUNKS=$NumChunks SAMPLE_FPS=$SampleFps N_LOCAL=$NLocal RETRIEVE_SIZE=$RetrieveSize bash scripts/run_remote_eval_template.sh"
    Run-Checked "ssh -o BatchMode=yes $RemoteHost `"$RemoteEvalCmd`""
}

Write-Host "`nResearch pipeline completed."
