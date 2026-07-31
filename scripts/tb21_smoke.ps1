# Terminal-Bench 2.1 smoke slice for nano-harness.
# Purpose: prove plumbing on the 2.1 dataset and see whether 2.1's resource/
# instruction fixes change nano-harness's result on a curated 10-task slice.
# This is NOT a score estimate and NOT a leaderboard submission (submission needs
# 5 trials/task and no timeout overrides).
#
# Per Codex handoff 008, this runs with NO timeout/resource overrides
# (--timeout-multiplier default 1.0) so it matches submission rules and reveals
# whether nano's loop terminates cleanly inside the official 2.1 limits.
# Errored trials are forced to reward ZERO on the official metric even if the
# verifier would pass the workspace, so clean termination is the whole game:
# score both raw verifier rewards AND submission-equivalent (errors zeroed).
#
# Usage (from repo root, fresh PowerShell):
#   .\scripts\tb21_smoke.ps1                 # default: mult 1.0 (submission-valid)
#   .\scripts\tb21_smoke.ps1 -TimeoutMult 2  # compare against the 2.0-style crutch
param(
    [string]$Model = "aws/claude4_8_opus",
    [double]$TimeoutMult = 1.0,   # 1.0 = official limits (no override). Codex: keep 1.0.
    [string]$JobName = ""
)

Set-Location $PSScriptRoot\..

# Load .env (API key + gateway URL) into this process only.
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
if (-not $env:OPENAI_API_KEY -and $env:ASU_AIML_TOKEN) { $env:OPENAI_API_KEY = $env:ASU_AIML_TOKEN }
# Derive the OpenAI-compatible base URL. The ASU gateway's OpenAI shim lives at
# /v1 (the native ASU endpoint stored in .env is /query, a different API shape).
if (-not $env:OPENAI_BASE_URL -and $env:ASU_AIML_ENDPOINT) {
    $env:OPENAI_BASE_URL = $env:ASU_AIML_ENDPOINT -replace '/query/?$', '/v1'
}
if (-not $env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL = "https://api.openai.com/v1" }
Write-Host "Using OPENAI_BASE_URL = $env:OPENAI_BASE_URL"
# Harbor writes result.json with the platform default encoding and crashes on
# unicode under Windows cp1252. Force UTF-8.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not $JobName) {
    $stamp = Get-Date -Format "MMdd-HHmm"
    $JobName = "nano-tb21-smoke-$stamp"
}

# Codex-curated slice (handoff 008): prior nano failures + high-2.1-delta tasks +
# unchanged controls. Not representative of the full 89; do not extrapolate a score.
$tasks = @(
    "caffe-cifar-10",          # prior AgentTimeout; 2.1 raised timeout/CPU/memory + nproc
    "make-doom-for-mips",      # prior AgentTimeout; 2.1 fixed stale package index
    "overfull-hbox",           # prior nonzero exit; 2.1 dependency-pin cleanup
    "qemu-startup",            # prior exit-137 OOM; negative control (may still OOM)
    "polyglot-c-py",           # large official 2.1 pass-rate gain
    "polyglot-rust-c",         # large official 2.1 pass-rate gain
    "torch-tensor-parallelism",# 2.1 instruction clarification + memory increase
    "mteb-retrieve",           # 2.1 dependency pin + instruction clarification
    "write-compressor",        # unchanged control (nano failed this in 2.0)
    "regex-log"                # unchanged lightweight control (nano passed in 2.0)
)
$harborArgs = @(
    'run',
    '-d', 'terminal-bench/terminal-bench-2-1',
    '-a', 'eval.tb_agent:NanoAgent',
    '-m', $Model
)
# Task names in this dataset are org-prefixed (e.g. terminal-bench/regex-log),
# and -i matches the full name, so prefix each filter.
foreach ($t in $tasks) { $harborArgs += @('-i', "terminal-bench/$t") }
$harborArgs += @(
    '-o', 'results\terminal-bench',
    '--job-name', $JobName,
    '--timeout-multiplier', $TimeoutMult,
    '-n', '2',
    '-y'
)

& .venv\Scripts\harbor.exe @harborArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nHarbor exited with code $LASTEXITCODE - the run FAILED, no valid results." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nSmoke slice done. Results: results\terminal-bench\$JobName\result.json"
Write-Host "Watch for: dataset resolved + all 10 containers started, adapter loaded,"
Write-Host "complete trial artifacts, and whether prior failures (caffe-cifar-10,"
Write-Host "make-doom-for-mips, overfull-hbox, qemu-startup) behave differently on 2.1."
