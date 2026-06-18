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

Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @(
        "run_rerun_queue.py",
        "--workspace-root", $workspace,
        "--python-bin", $pythonExe,
        "--cpu-only"
    ) `
    -WorkingDirectory $workspace `
    -WindowStyle Normal

Write-Host "Fila de reruns iniciada em CPU."
Write-Host "Workspace: $workspace"
Write-Host "Logs: $(Join-Path $workspace 'rerun_logs')"
