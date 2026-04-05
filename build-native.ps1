param(
    [switch]$Run
)

$ErrorActionPreference = "Stop"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio Build Tools first."
}

$installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installationPath) {
    throw "Visual Studio C++ Build Tools were not found."
}

$vsDev = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
$cmake = Join-Path $installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = Join-Path $installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$toolchain = Join-Path $installationPath "VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
$overlayTriplets = (Resolve-Path ".\vcpkg-triplets").Path

foreach ($path in @($vsDev, $cmake, $ninja, $toolchain, $overlayTriplets)) {
    if (-not (Test-Path $path)) {
        throw "Required tool was not found: $path"
    }
}

$configure = "`"$vsDev`" -arch=x64 && `"$cmake`" -S . -B build\native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_TOOLCHAIN_FILE=`"$toolchain`" -DVCPKG_OVERLAY_TRIPLETS=`"$overlayTriplets`" -DVCPKG_TARGET_TRIPLET=x64-windows-release"
$build = "`"$vsDev`" -arch=x64 && `"$cmake`" --build build\native --config Release"

Write-Host "Configuring native build..."
cmd /c $configure
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Building native app..."
cmd /c $build
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Run) {
    $exe = Join-Path (Resolve-Path "build\native").Path "local_face_filter.exe"
    if (-not (Test-Path $exe)) {
        throw "Build completed but executable was not found at $exe"
    }
    & $exe
}
