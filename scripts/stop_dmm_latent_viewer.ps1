param([int]$Port = 8877)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$recordPath = Join-Path $project ".runtime\dmm_latent_viewer\server-$Port.json"
if (-not (Test-Path -LiteralPath $recordPath)) { Write-Output 'No viewer process record.'; return }
$record = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
$viewer = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)"
if (-not $viewer) { Write-Output 'Viewer has already stopped.'; return }
$expected = Join-Path $project 'apps\dmm_latent_viewer\server.py'
if (-not $viewer.CommandLine.Contains($expected)) { throw 'Process identity changed; refusing to stop it.' }
Stop-Process -Id $record.pid
Write-Output "DMM viewer stopped on port $Port."
