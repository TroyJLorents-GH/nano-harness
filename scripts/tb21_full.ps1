# Full 89-task Terminal-Bench 2.1 run for nano-harness.
#
# One trial per task at 1.0x (official limits, no overrides). This is a
# research run, NOT a leaderboard submission: a valid submission needs >=5
# trials per task (445 trials), a public Harbor Hub upload, and a PR.
#
# Scoring note: the official metric forces an errored trial (AgentTimeoutError
# etc.) to reward 0 even when the verifier would pass the workspace. Score
# BOTH numbers: raw verifier reward, and submission-equivalent with every
# errored trial zeroed.
#
# Usage (from repo root, fresh PowerShell, Docker running):
#   .\scripts\tb21_full.ps1                     # Opus 4.8, no deadline flag
#   .\scripts\tb21_full.ps1 -UseDeadline        # + self-imposed clean exits
#   .\scripts\tb21_full.ps1 -Model aws/claude5_opus
#
# Expect ~16h wall clock at -n 2. Resume an interrupted job with:
#   .venv\Scripts\harbor.exe job resume --job-path results\terminal-bench\<job>
param(
    [string]$Model = "aws/claude4_8_opus",
    [double]$TimeoutMult = 1.0,   # keep 1.0: submission rules forbid overrides
    [int]$Concurrent = 2,
    [string]$JobName = "",
    [switch]$UseDeadline
)

Set-Location $PSScriptRoot\..

# Load .env (API key + gateway URL) into this process only.
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
if (-not $env:OPENAI_API_KEY -and $env:ASU_AIML_TOKEN) { $env:OPENAI_API_KEY = $env:ASU_AIML_TOKEN }
# The ASU gateway's OpenAI-compatible shim lives at /v1; the endpoint stored
# in .env is the native /query API, a different shape.
if (-not $env:OPENAI_BASE_URL -and $env:ASU_AIML_ENDPOINT) {
    $env:OPENAI_BASE_URL = $env:ASU_AIML_ENDPOINT -replace '/query/?$', '/v1'
}
if (-not $env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL = "https://api.openai.com/v1" }
# Harbor writes result.json with the platform default encoding and crashes on
# unicode under Windows cp1252. Force UTF-8.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# Probe-verified: the gateway accepts >=64k output tokens. 16384 halves the
# continuation round trips on large file writes.
$env:NANO_MAX_TOKENS = "16384"
# Deadline-aware clean exits read each task's timeout host-side in the
# adapter. Opt-in, because Harbor does not expose that deadline to agents by
# design - a number produced with this flag is a research number.
if ($UseDeadline) { $env:NANO_USE_DEADLINE = "1" } else { Remove-Item Env:NANO_USE_DEADLINE -ErrorAction SilentlyContinue }

Write-Host "Model:            $Model"
Write-Host "OPENAI_BASE_URL:  $env:OPENAI_BASE_URL"
Write-Host "Deadline flag:    $(if ($UseDeadline) { 'ON (research run)' } else { 'off (no caveat)' })"

if (-not $JobName) {
    $safe = $Model -replace '[/.]', '-'
    $stamp = Get-Date -Format "MMdd-HHmm"
    $JobName = "nano-tb21-full-$safe-$stamp"
}

$harborArgs = @(
    'run',
    '-d', 'terminal-bench/terminal-bench-2-1',
    '-a', 'eval.tb_agent:NanoAgent',
    '-m', $Model,
    '-o', 'results\terminal-bench',
    '--job-name', $JobName,
    '--timeout-multiplier', $TimeoutMult,
    '-n', $Concurrent,
    '-y'
)

& .venv\Scripts\harbor.exe @harborArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nHarbor exited with code $LASTEXITCODE - the run FAILED." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nDone. Results: results\terminal-bench\$JobName\result.json"
Write-Host "Score BOTH: raw verifier reward, and submission-equivalent (errors zeroed)."
