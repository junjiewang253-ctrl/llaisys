# build.ps1 - 用于修复 VS2026 + xmake 环境问题
$env:_CL_ = '/I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared" /std:c++17'
$env:_LINK_ = '/LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"'

# 执行 xmake 命令（支持传参，如 clean, -v 等）
xmake $args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 只在“构建”时自动部署 DLL：
# - 不带参数：xmake 默认就是 build
# - build/b：显式构建
$doDeploy = $false
if ($args.Count -eq 0) {
    $doDeploy = $true
} elseif ($args[0] -in @("b", "build")) {
    $doDeploy = $true
}

if ($doDeploy) {
    $root  = $PSScriptRoot
    $src   = Join-Path $root "build\windows\x64\release\llaisys.dll"
    $dstPy = Join-Path $root "python\llaisys\libllaisys\llaisys.dll"
    $dstBin = Join-Path $root "bin\llaisys.dll"

    if (!(Test-Path $src)) {
        Write-Host "[deploy] ERROR: not found: $src"
        Write-Host "[deploy] hint: check build mode (release) / arch (x64) or xmake output dir."
        exit 1
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $dstPy) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $dstBin) | Out-Null

    Copy-Item $src $dstPy -Force
    Copy-Item $src $dstBin -Force

    Write-Host "[deploy] copied llaisys.dll to:"
    Write-Host "  - $dstPy"
    Write-Host "  - $dstBin"
}