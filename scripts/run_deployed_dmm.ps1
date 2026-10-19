# Use the exported R4 models with their own compatible DMM source and Python environment.
# Examples:
#   .\scripts\run_deployed_dmm.ps1 --help
#   .\scripts\run_deployed_dmm.ps1 validate-scene D:\scenes\case_001\fit_input\manifest.json --device cpu
# Relative scene/config/output paths are resolved from the deployed package directory.
$ErrorActionPreference = 'Stop'
$deployment = Join-Path (Split-Path $PSScriptRoot -Parent) '.runtime\dmm_r4_best_val_20261015'
$python = Join-Path $deployment 'env\Scripts\python.exe'
$package = Join-Path $deployment 'package'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing deployment environment: $python" }
$saved = @{}
foreach ($key in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE')) {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Push-Location $package
    try {
        & $python -u -B (Join-Path $package 'third_party\DMM\dmm_cli.py') @args
        $dmmExit = $LASTEXITCODE
    } finally { Pop-Location }
} finally {
    foreach ($key in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process')
    }
}
if ($dmmExit -ne 0) { throw "DMM exited with status $dmmExit" }
