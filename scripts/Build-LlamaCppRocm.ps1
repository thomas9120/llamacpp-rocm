<#
.SYNOPSIS
    Build llama.cpp with AMD ROCm (TheRock) acceleration for a given GPU target.
    Defaults to gfx1151 (STX Halo / Ryzen AI MAX+ 395).

.DESCRIPTION
    This script reproduces the upstream lemonade-sdk/llamacpp-rocm GitHub Actions
    build pipeline (see .github/workflows/build-llamacpp-rocm.yml) so it can run
    locally on a Windows machine. It will:

      1. Verify / install required build tools (Ninja, Strawberry Perl, VS Build Tools).
      2. Download the matching TheRock ROCm 7 nightly tarball for your GPU target.
      3. Extract the ROCm SDK to C:\opt\rocm.
      4. Shallow-clone (or update) llama.cpp from https://github.com/ggerganov/llama.cpp.
      5. Configure + build with CMake/Ninja using ROCm's bundled clang (GGML_HIP=ON).
      6. Stage all required ROCm runtime DLLs and rocblas/hipblaslt kernels next to
         llama-server.exe so the result is self-contained and portable.

    The script is re-runnable: each stage is idempotent and can be skipped via flags.

.PARAMETER GpuTarget
    AMD GPU target group. Default: gfx1151.
    Accepted: gfx1151, gfx1150, gfx120X, gfx110X, gfx103X, gfx90a, gfx908.
    Groups (gfx120X/gfx110X/gfx103X) expand to multiple architectures.

.PARAMETER RocmVersion
    TheRock ROCm version to use, e.g. "7.13.0a20260318", or "latest" (default)
    to auto-detect the newest nightly from https://rocm.nightlies.amd.com/tarball-multi-arch.

.PARAMETER LlamacppVersion
    llama.cpp tag / branch / commit to build. Default: "latest" (master).

.PARAMETER RocmDir
    Where to install the ROCm SDK. Default: C:\opt\rocm (matches upstream).

.PARAMETER SourceDir
    Where to clone llama.cpp. Default: .\llama.cpp (next to this script).

.PARAMETER BuildDir
    Build output directory. Default: .\llama.cpp\build (matches upstream).

.PARAMETER StagingDir
    Final self-contained output directory. Default: .\build-output\<GpuTarget>.
    All llama-*.exe binaries + ROCm runtime DLLs + kernel catalogs land here.

.PARAMETER StrixHaloFaFix
    Quantized-KV flash-attention patch mode: auto, on, or off. Default: auto,
    which applies the patch only to gfx1151 builds.

.PARAMETER BuildTests
    Build llama.cpp test binaries, including test-backend-ops.

.PARAMETER BuildJobs
    Maximum parallel build jobs. Default: 0, which caps the local HIP build at
    8 jobs to avoid compiler oversubscription.

.PARAMETER EnableCcache
    Enable compiler caching. Disabled by default because clean Windows HIP builds
    can stall in the ccache frontend shipped with Strawberry Perl.

.PARAMETER SkipDeps
    Skip the dependency check/install stage.

.PARAMETER SkipRocmDownload
    Skip downloading/extracting ROCm (assume -RocmDir already populated).

.PARAMETER SkipClone
    Skip cloning/updating llama.cpp (assume -SourceDir already populated).

.PARAMETER SkipBuild
    Skip the CMake configure/build stage.

.PARAMETER SkipStage
    Skip copying runtime DLLs to the staging directory.

.PARAMETER EnableHipVmm
    Build with HIP VMM enabled by passing -DGGML_HIP_NO_VMM=OFF to CMake.
    This may help RDNA 4 / memory-pressure cases, but is opt-in because some
    gfx1151 setups report better stability with llama.cpp's default NO_VMM path.

.PARAMETER Clean
    Remove -SourceDir, -BuildDir, and -RocmDir before building (fresh build).

.EXAMPLE
    .\scripts\Build-LlamaCppRocm.ps1
    Builds llama.cpp + ROCm for gfx1151 using the latest nightlies.

.EXAMPLE
    .\scripts\Build-LlamaCppRocm.ps1 -GpuTarget gfx1151 -RocmVersion 7.13.0a20260318
    Builds against a specific TheRock ROCm version.

.EXAMPLE
    .\scripts\Build-LlamaCppRocm.ps1 -SkipRocmDownload -SkipClone -SkipDeps
    Re-run just the configure/build/stage stages (fastest iteration loop).

.EXAMPLE
    .\scripts\Build-LlamaCppRocm.ps1 -EnableHipVmm
    Builds an opt-in HIP VMM variant in separate build/output folders.

.NOTES
    * Requires Visual Studio 2022 Build Tools (VC++, ATL, Windows 11 SDK).
      Upstream explicitly pins to VS 2022 (MSVC 14.4x). VS 2026 (MSVC 14.51)
      currently breaks the HIP build due to a <cmath> constexpr collision —
      see .github/workflows/build-llamacpp-rocm.yml and llama.cpp#22570.
    * After building, run from -StagingDir:
        .\llama-server.exe -m <model.gguf> -ngl 99
#>

[CmdletBinding()]
param(
    [ValidateSet('gfx1151','gfx1150','gfx120X','gfx110X','gfx103X','gfx90a','gfx908')]
    [string]$GpuTarget = 'gfx1151',

    [string]$RocmVersion = 'latest',
    [string]$LlamacppVersion = 'latest',

    [string]$RocmDir     = 'C:\opt\rocm',
    [string]$SourceDir   = $null,
    [string]$BuildDir    = $null,                 # defaults to SourceDir\build
    [string]$StagingDir  = $null,                 # defaults to .\build-output\<GpuTarget>

    [ValidateSet('auto','on','off')]
    [string]$StrixHaloFaFix = 'auto',

    [ValidateRange(0, 256)]
    [int]$BuildJobs = 0,

    [switch]$SkipDeps,
    [switch]$SkipRocmDownload,
    [switch]$SkipClone,
    [switch]$SkipBuild,
    [switch]$SkipStage,
    [switch]$BuildTests,
    [switch]$EnableCcache,
    [switch]$EnableHipVmm,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$buildVariantSuffix = if ($EnableHipVmm) { '-hip-vmm' } else { '' }
if (-not $SourceDir) { $SourceDir = Join-Path $scriptRoot 'llama.cpp' }
if (-not $BuildDir)   { $BuildDir   = Join-Path $SourceDir "build$buildVariantSuffix" }
if (-not $StagingDir) { $StagingDir = Join-Path (Join-Path $scriptRoot 'build-output') "$GpuTarget$buildVariantSuffix" }

$applyStrixHaloFaFix = switch ($StrixHaloFaFix) {
    'on'      { $true }
    'off'     { $false }
    default   { $GpuTarget -eq 'gfx1151' }
}

$effectiveBuildJobs = if ($BuildJobs -gt 0) {
    $BuildJobs
} else {
    [Math]::Max(1, [Math]::Min([Environment]::ProcessorCount, 8))
}
$llamacppPatchset = 'none'
$patchScript = Join-Path $scriptRoot 'Apply-StrixHaloFaPatch.ps1'
$patchsetFile = Join-Path (Join-Path (Split-Path -Parent $scriptRoot) 'patches\strix-halo-fa') 'PATCHSET'
$strixHaloPatchset = (Get-Content -LiteralPath $patchsetFile -Raw).Trim()
$patchsetMarker = Join-Path $SourceDir '.llamacpp-rocm-build.patchset'

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

function Write-Step([string]$msg) { Write-Host "`n====[ $msg ]====" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[!]  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "[X]  $msg" -ForegroundColor Red }

function Test-Command([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Get-VsWherePath {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $null
    }
    return $vswhere
}

function Get-VisualStudioBuildToolsPath {
    param([Parameter(Mandatory)][string]$VsWhere)

    $required = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'

    # ROCm HIP currently needs the VS 2022 MSVC headers. VS 2026 / MSVC 14.5x
    # collides with clang's HIP math wrappers in <cmath>.
    $vs2022 = & $VsWhere -products * `
        -version '[17.0,18.0)' `
        -requires $required `
        -property installationPath
    if ($vs2022) {
        return ($vs2022 | Select-Object -First 1)
    }

    return & $VsWhere -latest -products * `
        -requires $required `
        -property installationPath
}

function Get-MsvcToolsetVersion([string]$vsPath) {
    $msvcRoot = Join-Path $vsPath 'VC\Tools\MSVC'
    if (-not (Test-Path $msvcRoot)) {
        return $null
    }

    $toolset = Get-ChildItem -Path $msvcRoot -Directory |
        Sort-Object { [version]$_.Name } -Descending |
        Select-Object -First 1
    return $toolset.Name
}

# TheRock uses generic group names for some targets. These need to be expanded
# for the cmake -DGPU_TARGETS flag and suffixed with "-all" for tarball lookups.
function Resolve-MappedTarget([string]$target) {
    switch ($target) {
        'gfx110X' { return 'gfx1100;gfx1101;gfx1102;gfx1103' }
        'gfx103X' { return 'gfx1030;gfx1031;gfx1032;gfx1034' }
        'gfx120X' { return 'gfx1200;gfx1201' }
        default   { return $target }   # gfx1151, gfx1150, gfx90a, gfx908
    }
}

function Resolve-TarballTarget([string]$target) {
    switch ($target) {
        'gfx103X' { return 'gfx103X-all' }
        'gfx110X' { return 'gfx110X-all' }
        'gfx120X' { return 'gfx120X-all' }
        default   { return $target }
    }
}

# ----------------------------------------------------------------------------- #
# Stage 0 - optional clean
# ----------------------------------------------------------------------------- #

if ($Clean) {
    Write-Step "Clean: removing previous sources, build, and ROCm SDK"
    foreach ($p in @($SourceDir, $BuildDir)) {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p; Write-Ok "Removed $p" }
    }
    if (Test-Path $RocmDir) {
        Remove-Item -Recurse -Force $RocmDir
        Write-Ok "Removed $RocmDir"
    }
}

# ----------------------------------------------------------------------------- #
# Stage 1 - dependencies
# ----------------------------------------------------------------------------- #

if (-not $SkipDeps) {
    Write-Step "Stage 1/5 - verify build dependencies"

    # --- Visual Studio Build Tools (VC++ + ATL + Win11 SDK) ------------------- #
    $vswhere = Get-VsWherePath
    if (-not $vswhere) {
        Write-Err "vswhere.exe not found. Install Visual Studio 2022 Build Tools:"
        Write-Err "  choco install visualstudio2022buildtools -y --params `"--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621`""
        throw "Visual Studio Build Tools not installed"
    }
    $vsPath = Get-VisualStudioBuildToolsPath -VsWhere $vswhere
    if (-not $vsPath) {
        Write-Err "No Visual Studio installation with VC++ Tools found."
        throw "VC++ Tools not installed"
    }
    # Detect the MSVC toolset version and warn if it is the known-broken 14.51.
    $msvcVer = Get-MsvcToolsetVersion $vsPath
    Write-Ok "Visual Studio: $vsPath (MSVC $msvcVer)"
    if ($msvcVer -match '^14\.5') {
        Write-Warn "VS 2026 (MSVC 14.5x) detected. Upstream pins to VS 2022 because"
        Write-Warn "MSVC 14.51 <cmath> collides with ROCm clang HIP math wrappers"
        Write-Warn "(llama.cpp#22570). If the build fails in HIP math headers, install"
        Write-Warn "the VS 2022 Build Tools side-by-side and re-run."
    }

    # --- Ninja ---------------------------------------------------------------- #
    if (-not (Test-Command 'ninja')) {
        Write-Warn "ninja not on PATH. Install with: choco install ninja -y  (or winget install Ninja-build.Ninja)"
        throw "ninja required"
    }
    Write-Ok "ninja: $((Get-Command ninja).Source)"

    # --- CMake ---------------------------------------------------------------- #
    if (-not (Test-Command 'cmake')) {
        Write-Warn "cmake not on PATH. Install with: choco install cmake --version=3.31.0 -y"
        throw "cmake required"
    }
    Write-Ok "cmake: $((Get-Command cmake).Source)"

    # --- Strawberry Perl (HIP build scripts need it) -------------------------- #
    if (-not (Test-Command 'perl')) {
        Write-Warn "perl not on PATH. Installing Strawberry Perl silently..."
        $perlUrl  = 'http://strawberryperl.com/download/5.32.1.1/strawberry-perl-5.32.1.1-64bit.msi'
        $perlMsi  = Join-Path $env:TEMP 'strawberry-perl-5.32.1.1-64bit.msi'
        Invoke-WebRequest -Uri $perlUrl -OutFile $perlMsi
        Start-Process msiexec.exe -ArgumentList "/i `"$perlMsi`" /quiet /norestart" -Wait
        Remove-Item $perlMsi -Force
        # Strawberry Perl puts perl in C:\Strawberry\perl\bin
        $env:PATH = "C:\Strawberry\perl\bin;$env:PATH"
        if (-not (Test-Command 'perl')) {
            throw "Strawberry Perl install failed - install manually and re-run with -SkipDeps"
        }
    }
    Write-Ok "perl: $((Get-Command perl).Source)"

    # --- git ------------------------------------------------------------------ #
    if (-not (Test-Command 'git')) { throw "git required (install from https://git-scm.com)" }
    Write-Ok "git: $((Get-Command git).Source)"
}

# ----------------------------------------------------------------------------- #
# Stage 2 - download + extract TheRock ROCm SDK
# ----------------------------------------------------------------------------- #

$rocmClang     = Join-Path $RocmDir 'lib\llvm\bin\clang.exe'
$rocmClangPlus = Join-Path $RocmDir 'lib\llvm\bin\clang++.exe'

if (-not $SkipRocmDownload) {
    Write-Step "Stage 2/5 - download + extract TheRock ROCm SDK to $RocmDir"

    $tarballTarget = Resolve-TarballTarget $GpuTarget
    $baseIndexUrl  = 'https://rocm.nightlies.amd.com/tarball-multi-arch'

    if ($RocmVersion -eq 'latest') {
        Write-Host "Auto-detecting latest ROCm nightly for target: $tarballTarget"
        $indexHtml = (Invoke-WebRequest "$baseIndexUrl/" -UseBasicParsing).Content
        $filesMatch = [regex]::Match(
            $indexHtml,
            'const files = (\[.*?\]);',
            [System.Text.RegularExpressions.RegexOptions]::Singleline
        )
        if (-not $filesMatch.Success) {
            throw "Failed to parse file index at $baseIndexUrl/"
        }
        $allFiles  = $filesMatch.Groups[1].Value | ConvertFrom-Json
        $prefix    = "therock-dist-windows-$tarballTarget-"
        $latest    = $allFiles |
            Where-Object { $_.name -like "$prefix*" -and $_.name -match '\d{8}\.tar\.gz$' } |
            Sort-Object { [regex]::Match($_.name, '(\d{8})\.tar\.gz$').Groups[1].Value } |
            Select-Object -Last 1
        if (-not $latest) {
            throw "No tarball found for prefix '$prefix' at $baseIndexUrl/"
        }
        $rocmFile    = $latest.name
        $rocmVersion = if ($rocmFile -match "therock-dist-windows-$tarballTarget-(\d+\.\d+\.\d+(?:a|rc)\d+)\.tar\.gz") {
            $matches[1]
        } else { throw "Could not parse ROCm version from $rocmFile" }
        Write-Ok "Latest ROCm nightly: $rocmFile (version $rocmVersion)"
    } else {
        $rocmVersion = $RocmVersion
        $rocmFile    = "therock-dist-windows-$tarballTarget-$rocmVersion.tar.gz"
    }

    # Re-use already-extracted SDK if the version matches.
    $versionMarker = Join-Path $RocmDir '.llamacpp-rocm-build.version'
    if ((Test-Path $rocmClang) -and (Test-Path $versionMarker) `
        -and (Get-Content $versionMarker -Raw) -eq $rocmVersion) {
        Write-Ok "ROCm SDK $rocmVersion already extracted at $RocmDir - skipping download"
    } else {
        $rocmUrl = "$baseIndexUrl/$rocmFile"
        $tarPath = Join-Path $env:TEMP $rocmFile
        Write-Host "Downloading: $rocmUrl"
        Write-Host "To:           $tarPath"
        Invoke-WebRequest -Uri $rocmUrl -OutFile $tarPath

        # Fresh extract: clear any partial prior install.
        if (Test-Path $RocmDir) { Remove-Item -Recurse -Force $RocmDir }
        New-Item -ItemType Directory -Force -Path $RocmDir | Out-Null

        Write-Host "Extracting to $RocmDir (this takes a few minutes)..."
        # tar.exe is present on Windows 10+ and handles .tar.gz natively.
        & tar -xzf $tarPath -C $RocmDir --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "tar extraction failed (exit $LASTEXITCODE)" }

        Set-Content -Path $versionMarker -Value $rocmVersion -NoNewline
        Remove-Item $tarPath -Force
        Write-Ok "ROCm SDK $rocmVersion extracted to $RocmDir"
    }
} else {
    Write-Step "Stage 2/5 - SKIPPED (assuming ROCm SDK at $RocmDir)"
}

if (-not (Test-Path $rocmClang)) {
    throw "ROCm clang not found at $rocmClang. Run without -SkipRocmDownload or point -RocmDir at a valid SDK."
}

# ----------------------------------------------------------------------------- #
# Stage 3 - clone / update llama.cpp
# ----------------------------------------------------------------------------- #

if (-not $SkipClone) {
    Write-Step "Stage 3/5 - clone / update llama.cpp"

    if ($LlamacppVersion -eq 'latest') {
        $ref = 'master'
    } else {
        $ref = $LlamacppVersion
    }

    if (Test-Path (Join-Path $SourceDir '.git')) {
        Write-Host "Existing clone at $SourceDir - fetching + resetting to $ref"
        & git -C $SourceDir fetch --depth 1 origin $ref
        if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }
        & git -C $SourceDir checkout $ref
        & git -C $SourceDir reset --hard "origin/$ref"
    } else {
        Write-Host "Cloning llama.cpp ($ref) into $SourceDir"
        & git clone --depth 1 --single-branch --branch $ref https://github.com/ggerganov/llama.cpp.git $SourceDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }

    $commitHash = (& git -C $SourceDir rev-parse --short=5 HEAD).Trim()
    Write-Ok "llama.cpp @ $commitHash ($ref)"
    Set-Content -Path (Join-Path $SourceDir '.llamacpp-rocm-build.commit') -Value $commitHash -NoNewline
} else {
    Write-Step "Stage 3/5 - SKIPPED (using existing llama.cpp at $SourceDir)"
}

if ($applyStrixHaloFaFix) {
    Write-Step "Stage 3b/5 - apply gfx1151 quantized-KV flash-attention patch"
    & $patchScript -SourceDir $SourceDir
    $llamacppPatchset = $strixHaloPatchset
} else {
    $previousPatchset = if ($SkipClone -and (Test-Path -LiteralPath $patchsetMarker -PathType Leaf)) {
        (Get-Content -LiteralPath $patchsetMarker -Raw).Trim()
    } else {
        'none'
    }
    if ($previousPatchset -eq $strixHaloPatchset) {
        Write-Step "Stage 3b/5 - remove gfx1151 quantized-KV flash-attention patch"
        & $patchScript -SourceDir $SourceDir -Remove
    }
    Write-Host "Strix Halo flash-attention patch disabled (mode=$StrixHaloFaFix, target=$GpuTarget)."
}
Set-Content -Path $patchsetMarker -Value $llamacppPatchset -NoNewline

# ----------------------------------------------------------------------------- #
# Stage 4 - configure + build (CMake + Ninja under the VS dev shell)
# ----------------------------------------------------------------------------- #

$mappedTarget = Resolve-MappedTarget $GpuTarget
$hipVmmCmakeLine = if ($EnableHipVmm) { "  -DGGML_HIP_NO_VMM=OFF ^`r`n" } else { "" }
$buildTestsCmakeValue = if ($BuildTests) { 'ON' } else { 'OFF' }
$ccacheCmakeValue = if ($EnableCcache) { 'ON' } else { 'OFF' }

if (-not $SkipBuild) {
    Write-Step "Stage 4/5 - configure + build (GPU_TARGETS=$mappedTarget, jobs=$effectiveBuildJobs)"
    if ($EnableHipVmm) {
        Write-Warn "HIP VMM enabled: passing -DGGML_HIP_NO_VMM=OFF to CMake."
    }

    # Clean prior build dir so re-configures don't pick up stale cache.
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    # Locate vcvars64.bat for the VS dev environment. The HIP/clang toolchain
    # needs the MSVC headers + libs + Win SDK on PATH/INCLUDE/LIB, so we wrap
    # the whole cmake invocation in a cmd script that sources vcvars64 first.
    $vswhere = Get-VsWherePath
    if (-not $vswhere) { throw "vswhere.exe not found. Install Visual Studio 2022 Build Tools." }
    $vsPath  = Get-VisualStudioBuildToolsPath -VsWhere $vswhere
    if (-not $vsPath) { throw "No Visual Studio installation with VC++ Tools found." }
    $msvcVer = Get-MsvcToolsetVersion $vsPath
    if ($msvcVer -match '^14\.5') {
        Write-Warn "Using MSVC $msvcVer. ROCm HIP builds are known to fail with MSVC 14.5x."
        Write-Warn "Install Visual Studio 2022 Build Tools and re-run if HIP math headers fail."
    } else {
        Write-Ok "Using Visual Studio toolset: $vsPath (MSVC $msvcVer)"
    }
    $vcvars  = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    # Temp cmd wrapper - we need vcvars + HIP env all in the same process.
    $bat = Join-Path $env:TEMP "llamacpp-rocm-build-$([guid]::NewGuid().ToString('N')).cmd"
    @"
@echo off
call `"$vcvars`" || exit /b 1
set `"HIP_PATH=$RocmDir`"
set `"HIP_PLATFORM=amd`"
set `"PATH=$RocmDir\lib\llvm\bin;$RocmDir\bin;%PATH%`"
cd /d `"$BuildDir`" || exit /b 1
cmake `"$SourceDir`" -G Ninja ^
  -DCMAKE_C_COMPILER=`"$rocmClang`" ^
  -DCMAKE_CXX_COMPILER=`"$rocmClangPlus`" ^
  -DCMAKE_CXX_FLAGS=`"-I$RocmDir\include`" ^
  -DCMAKE_CROSSCOMPILING=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGPU_TARGETS=`"$mappedTarget`" ^
  -DBUILD_SHARED_LIBS=ON ^
  -DLLAMA_BUILD_TESTS=$buildTestsCmakeValue ^
  -DGGML_CCACHE=$ccacheCmakeValue ^
  -DGGML_HIP=ON ^
$hipVmmCmakeLine  -DGGML_OPENMP=OFF ^
  -DGGML_CUDA_FORCE_CUBLAS=OFF ^
  -DGGML_HIP_ROCWMMA_FATTN=OFF ^
  -DLLAMA_CURL=OFF ^
  -DGGML_NATIVE=OFF ^
  -DGGML_STATIC=OFF ^
  -DCMAKE_SYSTEM_NAME=Windows
@if errorlevel 1 exit /b 1
cmake --build . -j $effectiveBuildJobs
@exit /b %errorlevel%
"@ | Set-Content -Path $bat -Encoding ASCII

    Write-Host "Configuring + building via: $bat"
    & cmd /c $bat
    $buildExit = $LASTEXITCODE
    Remove-Item $bat -Force -ErrorAction SilentlyContinue
    if ($buildExit -ne 0) { throw "Build failed (exit $buildExit)" }

    $builtBin = Join-Path $BuildDir 'bin'
    Write-Ok "Build complete. Artifacts in $builtBin"
} else {
    Write-Step "Stage 4/5 - SKIPPED (assuming build at $BuildDir)"
}

# ----------------------------------------------------------------------------- #
# Stage 5 - stage binaries + ROCm runtime into a portable folder
# ----------------------------------------------------------------------------- #

if (-not $SkipStage) {
    Write-Step "Stage 5/5 - stage self-contained output at $StagingDir"

    $builtBin     = Join-Path $BuildDir 'bin'
    $rocmBinPath  = Join-Path $RocmDir 'bin'

    if (-not (Test-Path $builtBin)) {
        throw "Build output not found at $builtBin. Run without -SkipBuild."
    }

    if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
    New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

    # --- llama.cpp binaries + built DLLs -------------------------------------- #
    Write-Host "Copying llama.cpp build output..."
    Copy-Item -Path (Join-Path $builtBin '*') -Destination $StagingDir -Recurse -Force
    Set-Content -Path (Join-Path $StagingDir 'llamacpp-rocm-patchset.txt') -Value $llamacppPatchset

    # --- ROCm core runtime DLLs ---------------------------------------------- #
    # Matches the upstream "Copy ROCm core DLLs" workflow step.
    Write-Host "Copying ROCm runtime DLLs..."
    $dllPatterns = @(
        'amdhip64_*.dll',
        'rocm_kpack.dll',
        'amd_comgr*.dll',
        'libhipblas.dll',
        'rocblas.dll',
        'rocsolver.dll',
        'hipblaslt.dll',
        'libhipblaslt.dll',
        'hipblas.dll'
    )
    foreach ($pattern in $dllPatterns) {
        $matching = Get-ChildItem -Path $rocmBinPath -Filter $pattern -ErrorAction SilentlyContinue
        foreach ($f in $matching) {
            Copy-Item $f.FullName -Destination $StagingDir -Force
            Write-Host "  + $($f.Name)"
        }
    }

    # --- rocblas kernel catalog ---------------------------------------------- #
    $rocblasSrc = Join-Path $rocmBinPath 'rocblas\library'
    if (Test-Path $rocblasSrc) {
        $rocblasDst = Join-Path $StagingDir 'rocblas\library'
        Copy-Item -Path $rocblasSrc -Destination $rocblasDst -Recurse -Force
        Write-Ok "Copied rocblas\library ($((Get-ChildItem $rocblasDst).Count) files)"
    } else {
        Write-Warn "rocblas\library not found at $rocblasSrc"
    }

    # --- hipblaslt kernel catalog -------------------------------------------- #
    $hipblasltSrc = Join-Path $rocmBinPath 'hipblaslt\library'
    if (Test-Path $hipblasltSrc) {
        $hipblasltDst = Join-Path $StagingDir 'hipblaslt\library'
        Copy-Item -Path $hipblasltSrc -Destination $hipblasltDst -Recurse -Force
        Write-Ok "Copied hipblaslt\library ($((Get-ChildItem $hipblasltDst).Count) files)"
    } else {
        Write-Warn "hipblaslt\library not found at $hipblasltSrc"
    }

    Write-Ok "Staging complete."
    Write-Host ""
    Write-Host "Smoketest with:" -ForegroundColor White
    Write-Host "    cd `"$StagingDir`"" -ForegroundColor White
    Write-Host "    .\llama-server.exe -m <your-model.gguf> -ngl 99" -ForegroundColor White
} else {
    Write-Step "Stage 5/5 - SKIPPED"
}

Write-Host ""
Write-Ok "All done."
