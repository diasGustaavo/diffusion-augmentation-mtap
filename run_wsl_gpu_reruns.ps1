$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workspace

$logsRoot = Join-Path $workspace "rerun_logs"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

$drive = $workspace.Substring(0, 1).ToLowerInvariant()
$rest = $workspace.Substring(2) -replace "\\", "/"
$workspaceWsl = "/mnt/$drive$rest"

$bashCommand = @"
cd "$workspaceWsl"
chmod +x ./run_wsl_gpu_reruns.sh
./run_wsl_gpu_reruns.sh >> rerun_logs/wsl_gpu_reruns.out 2>&1
"@

Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList @("bash", "-lc", $bashCommand) `
    -WorkingDirectory $workspace `
    -WindowStyle Hidden

Write-Host "Rerun queue started on WSL with a dedicated launcher."
Write-Host "Workspace WSL: $workspaceWsl"
Write-Host "Log: $(Join-Path $logsRoot 'wsl_gpu_reruns.out')"
