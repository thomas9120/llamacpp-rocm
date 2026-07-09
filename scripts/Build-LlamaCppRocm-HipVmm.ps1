<#
.SYNOPSIS
    Build the opt-in HIP VMM variant of llama.cpp ROCm.

.DESCRIPTION
    This wrapper keeps the standard build script as the source of truth while
    enabling llama.cpp's HIP VMM path with -DGGML_HIP_NO_VMM=OFF.

    Output defaults to:
      scripts\llama.cpp\build-hip-vmm
      scripts\build-output\<GpuTarget>-hip-vmm

    Use this alongside the regular build so both variants can be compared.
#>

[CmdletBinding()]
param(
    [ValidateSet('gfx1151','gfx1150','gfx120X','gfx110X','gfx103X','gfx90a','gfx908')]
    [string]$GpuTarget = 'gfx1151',

    [string]$RocmVersion = 'latest',
    [string]$LlamacppVersion = 'latest',

    [string]$RocmDir     = 'C:\opt\rocm',
    [string]$SourceDir   = $null,
    [string]$BuildDir    = $null,
    [string]$StagingDir  = $null,

    [switch]$SkipDeps,
    [switch]$SkipRocmDownload,
    [switch]$SkipClone,
    [switch]$SkipBuild,
    [switch]$SkipStage,
    [switch]$Clean
)

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $SourceDir) { $SourceDir = Join-Path $scriptRoot 'llama.cpp' }

$paramsToForward = @{
    GpuTarget       = $GpuTarget
    RocmVersion     = $RocmVersion
    LlamacppVersion = $LlamacppVersion
    RocmDir         = $RocmDir
    SourceDir       = $SourceDir
    EnableHipVmm    = $true
}

if ($BuildDir)   { $paramsToForward.BuildDir = $BuildDir }
if ($StagingDir) { $paramsToForward.StagingDir = $StagingDir }
if ($SkipDeps) { $paramsToForward.SkipDeps = $true }
if ($SkipRocmDownload) { $paramsToForward.SkipRocmDownload = $true }
if ($SkipClone) { $paramsToForward.SkipClone = $true }
if ($SkipBuild) { $paramsToForward.SkipBuild = $true }
if ($SkipStage) { $paramsToForward.SkipStage = $true }
if ($Clean) { $paramsToForward.Clean = $true }

& (Join-Path $scriptRoot 'Build-LlamaCppRocm.ps1') @paramsToForward
exit $LASTEXITCODE
