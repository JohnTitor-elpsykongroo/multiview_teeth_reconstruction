param([int]$Port = 8877, [ValidateSet(128,160,192,256)][int]$Resolution = 160)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$python = Join-Path $project '.runtime\dmm_r4_best_val_20261015\env\Scripts\python.exe'
$server = Join-Path $project 'apps\dmm_latent_viewer\server.py'
$logRoot = Join-Path $project '.runtime\dmm_latent_viewer'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$url = "http://127.0.0.1:$Port"
try {
    $existing = Invoke-RestMethod "$url/api/status" -TimeoutSec 3
    if ($existing.app -eq 'dmm-latent-viewer') {
        Write-Output "Viewer already running: $url"
        return
    }
} catch {}
$arguments = @('-u', '-B', "`"$server`"", '--port', "$Port", '--resolution', "$Resolution")
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot "server-$Port.log") -RedirectStandardError (Join-Path $logRoot "server-$Port.error.log")
@{ pid = $process.Id; port = $Port; resolution = $Resolution; url = $url; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logRoot "server-$Port.json") -Encoding UTF8
Write-Output "Viewer starting: $url"
Write-Output "First start builds component caches; subsequent starts reuse them. Logs: $logRoot"
