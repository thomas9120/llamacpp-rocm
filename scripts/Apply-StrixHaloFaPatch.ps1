<#
.SYNOPSIS
    Apply or remove the gfx1151 quantized-KV flash-attention patch set.

.PARAMETER SourceDir
    Path to a llama.cpp Git checkout.

.PARAMETER CheckOnly
    Verify that the patch set can be applied, without changing the checkout.

.PARAMETER Remove
    Remove an applied patch set. This is also idempotent when the source is
    compatible with the patch set but already unpatched.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$SourceDir,

    [switch]$CheckOnly,

    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot = Split-Path -Parent $scriptRoot
$patchRoot = Join-Path (Join-Path $repoRoot 'patches') 'strix-halo-fa'
$patchsetFile = Join-Path $patchRoot 'PATCHSET'

if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    throw "llama.cpp source directory not found: $SourceDir"
}
if (-not (Test-Path -LiteralPath (Join-Path $SourceDir '.git'))) {
    throw "Not a llama.cpp Git checkout: $SourceDir"
}
if (-not (Test-Path -LiteralPath $patchsetFile -PathType Leaf)) {
    throw "Patch-set identifier not found: $patchsetFile"
}

$resolvedSource = (Resolve-Path -LiteralPath $SourceDir).Path
$patchFiles = @(Get-ChildItem -LiteralPath $patchRoot -Filter '*.patch' -File | Sort-Object Name)
if ($patchFiles.Count -eq 0) {
    throw "No patch files found in $patchRoot"
}

$patchset = (Get-Content -LiteralPath $patchsetFile -Raw).Trim()

function Invoke-GitApplyCheck {
    param(
        [Parameter(Mandatory)]
        [System.IO.FileInfo[]]$Files,

        [switch]$Reverse
    )

    $arguments = @('-c', "safe.directory=$resolvedSource", '-C', $resolvedSource, 'apply', '--check', '--whitespace=error-all')
    if ($Reverse) {
        $arguments += '--reverse'
    }
    $arguments += @($Files | ForEach-Object { $_.FullName })

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = @(& git @arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

$reverseFiles = @($patchFiles | Sort-Object Name -Descending)
$reverseCheck = Invoke-GitApplyCheck -Files $reverseFiles -Reverse
if ($Remove) {
    if ($reverseCheck.ExitCode -ne 0) {
        $forwardCheck = Invoke-GitApplyCheck -Files $patchFiles
        if ($forwardCheck.ExitCode -eq 0) {
            Write-Host "[OK] $patchset is already absent from $resolvedSource"
            return
        }

        $details = (@($reverseCheck.Output) + @($forwardCheck.Output) | Out-String).Trim()
        throw "$patchset cannot be cleanly removed from $resolvedSource because the source is incompatible or partially patched.`n$details"
    }

    if ($CheckOnly) {
        Write-Host "[OK] $patchset can be removed cleanly from $resolvedSource"
        return
    }

    $arguments = @('-c', "safe.directory=$resolvedSource", '-C', $resolvedSource, 'apply', '--reverse', '--whitespace=error-all')
    $arguments += @($reverseFiles | ForEach-Object { $_.FullName })
    & git @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove $patchset from $resolvedSource"
    }

    Write-Host "[OK] Removed $patchset from $resolvedSource"
    return
}

if ($reverseCheck.ExitCode -eq 0) {
    Write-Host "[OK] $patchset is already applied to $resolvedSource"
    return
}

$forwardCheck = Invoke-GitApplyCheck -Files $patchFiles
if ($forwardCheck.ExitCode -ne 0) {
    $details = ($forwardCheck.Output | Out-String).Trim()
    throw "$patchset is incompatible with, or partially applied to, $resolvedSource.`n$details"
}

if ($CheckOnly) {
    Write-Host "[OK] $patchset can be applied cleanly to $resolvedSource"
    return
}

$arguments = @('-c', "safe.directory=$resolvedSource", '-C', $resolvedSource, 'apply', '--whitespace=error-all')
$arguments += @($patchFiles | ForEach-Object { $_.FullName })
& git @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply $patchset to $resolvedSource"
}

Write-Host "[OK] Applied $patchset to $resolvedSource"
