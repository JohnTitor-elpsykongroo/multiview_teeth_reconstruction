$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gpu_env.ps1"
& 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation -NoLogo
$env:INCLUDE += ';D:\WorkSpace\conda\envs\3dteethland\Library\include'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
# The host Python launcher rewrites PATH at startup; restore the explicit build PATH inside Python.
$env:DMM_BUILD_PATH = $env:PATH
$env:VSLANG = '1033'
Push-Location "$gpuProjectRoot\third_party\nvdiffrast"
try {
    & $gpuPython -c "import os,runpy; os.environ['PATH']=os.environ['DMM_BUILD_PATH']; import torch.utils.cpp_extension as c; c.SUBPROCESS_DECODE_ARGS=('utf-8','replace'); runpy.run_path('setup.py',run_name='__main__')" build_ext --build-lib "$gpuProjectRoot\.runtime\gpu-python" dist_info --output-dir "$gpuProjectRoot\.runtime\gpu-python"
    if ($LASTEXITCODE -ne 0) { throw 'nvdiffrast build failed' }
    Copy-Item -LiteralPath 'nvdiffrast' -Destination "$gpuProjectRoot\.runtime\gpu-python" -Recurse -Force
    Copy-Item -LiteralPath 'LICENSE.txt' -Destination "$gpuProjectRoot\.runtime\gpu-python\nvdiffrast\LICENSE.txt"
} finally { Pop-Location }
