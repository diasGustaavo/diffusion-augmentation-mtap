$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workspace

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $pythonLauncher) {
    throw "Nao encontrei o launcher 'py'. Instale o Python 3 no Windows."
}

$venvPath = Join-Path $workspace ".venv_tf_win"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $pythonExe)) {
    py -3.11 -m venv $venvPath
}

& $pythonExe -m pip install --upgrade pip setuptools wheel
& $pipExe install -r (Join-Path $workspace "requirements_tf.txt")

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:TF_XLA_FLAGS = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"

$supervisorLog = Join-Path $workspace "run_all_experiments_supervisor_windows.log"
$childLog = Join-Path $workspace "run_all_experiments_child_windows.log"

Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @(
        "run_all_experiments_supervisor.py",
        "--workspace-root", $workspace,
        "--cpu-only",
        "--child-log-path", $childLog,
        "--supervisor-log-path", $supervisorLog
    ) `
    -WorkingDirectory $workspace `
    -WindowStyle Normal

Write-Host "Supervisor iniciado em CPU."
Write-Host "Workspace: $workspace"
Write-Host "Log supervisor: $supervisorLog"
Write-Host "Log treino: $childLog"
