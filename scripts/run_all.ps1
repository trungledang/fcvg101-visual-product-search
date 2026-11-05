$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing local environment: $python" }
if (-not (Test-Path -LiteralPath 'data/inshop/Eval/list_eval_partition.txt')) {
    throw 'Dataset is not ready. Put the official partition at data/inshop/Eval/list_eval_partition.txt and extract Img/img.zip into data/inshop/img/.'
}

function Invoke-PilotStep([string[]]$Arguments) {
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pilot step failed (exit $LASTEXITCODE): $Arguments" }
}

Invoke-PilotStep @('-m', 'pytest', '-q')
Invoke-PilotStep @('scripts/pilot.py', 'preflight')
Invoke-PilotStep @('scripts/pilot.py', 'benchmark', '--n', '256')
Invoke-PilotStep @('scripts/pilot.py', 'audit', '--workers', '4')
# The local RTX 3050 Ti has 4 GB VRAM; batch 8 keeps inference well below its limit.
Invoke-PilotStep @('scripts/pilot.py', 'embed', '--batch-size', '8', '--workers', '2', '--threads', '4')
Invoke-PilotStep @('scripts/pilot.py', 'evaluate')
Invoke-PilotStep @('scripts/pilot.py', 'examples', '--per-group', '10')
