# Dot-source this file for the project-owned GPU extension and dependencies.
$gpuProjectRoot = Split-Path $PSScriptRoot -Parent
$env:CUDA_HOME = Join-Path $gpuProjectRoot '.runtime\cuda-13.2.1'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PYTHONPATH = (Join-Path $gpuProjectRoot '.runtime\gpu-python')
$env:PATH = (Join-Path $env:CUDA_HOME 'bin') + ';' + (Join-Path $gpuProjectRoot '.runtime\gpu-python\bin') + ';' + $env:PATH
$env:TORCH_EXTENSIONS_DIR = Join-Path $gpuProjectRoot '.runtime\torch-extensions'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '2'
$env:PYTHONDONTWRITEBYTECODE = '1'
$gpuPython = 'D:\WorkSpace\conda\envs\teeth-yolo\python.exe'
