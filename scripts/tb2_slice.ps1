# Run a Terminal-Bench 2.0 slice with nano-harness through the ASU gateway.
# Usage (from repo root, in a normal PowerShell window):
#   .\scripts\tb2_slice.ps1                # 10 tasks, haiku
#   .\scripts\tb2_slice.ps1 -NTasks 1      # single-task smoke
#   .\scripts\tb2_slice.ps1 -Model openai/gpt5_4_thinking -NTasks 10
#   .\scripts\tb2_slice.ps1 -Resume nano-tb2-haiku-10   # continue an interrupted job
param(
    [string]$Model = "aws/claude4_8_opus",
    [int]$NTasks = 10,
    [double]$AgentTimeoutMult = 2.0,  # match the 100-iteration budget; 1.0 = TB2 default clock
    [string]$JobName = "",
    [string]$Resume = "",
    [switch]$RetryErrors
)

Set-Location $PSScriptRoot\..

# Load .env (API token + gateway URL) into this process only. Put these in .env:
#   OPENAI_API_KEY=<your key>
#   OPENAI_BASE_URL=<your OpenAI-compatible endpoint, e.g. https://api.openai.com/v1>
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
# The ASU gateway token and endpoint travel together. An OPENAI_API_KEY
# inherited from the user environment (a real OpenAI key for some other
# project) is never valid against the ASU gateway: every trial 403s at
# iteration 1 with zero tokens used. When .env carries ASU credentials they
# win over anything inherited; the source is printed so this is visible.
if ($env:ASU_AIML_TOKEN -and $env:ASU_AIML_ENDPOINT) {
    $env:OPENAI_API_KEY  = $env:ASU_AIML_TOKEN
    $env:OPENAI_BASE_URL = $env:ASU_AIML_ENDPOINT -replace '/query/?$', '/v1'
}
if (-not $env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL = "https://api.openai.com/v1" }
Write-Host "API key source:   $(if ($env:ASU_AIML_TOKEN -and $env:OPENAI_API_KEY -eq $env:ASU_AIML_TOKEN) { 'ASU_AIML_TOKEN (.env)' } else { 'inherited OPENAI_API_KEY' })"
# Force UTF-8 for all Python file writes. Harbor writes trial result.json with
# pathlib.write_text(), which defaults to Windows cp1252 and crashes the whole
# job on any unicode in a trial result. This makes those writes UTF-8 safe.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if ($Resume) {
    $args = @('job','resume','--job-path',"results\terminal-bench\$Resume")
    if ($RetryErrors) {
        # Re-run trials that failed on infra/transient causes (storm-slowed
        # timeouts, container-start hiccups, interruptions), not model errors.
        foreach ($e in 'AgentTimeoutError','EnvironmentStartTimeoutError','CancelledError') {
            $args += @('--filter-error-type', $e)
        }
    }
    & .venv\Scripts\harbor.exe @args
    exit $LASTEXITCODE
}

if (-not $JobName) {
    $safe = $Model -replace '[/.]', '-'
    $stamp = Get-Date -Format "MMdd-HHmm"
    $JobName = "nano-tb2-$safe-$NTasks-$stamp"
}

& .venv\Scripts\harbor.exe run `
    -d terminal-bench@2.0 `
    -a eval.tb_agent:NanoAgent `
    -m $Model `
    -l $NTasks `
    -o results\terminal-bench `
    --job-name $JobName `
    --agent-timeout-multiplier $AgentTimeoutMult `
    -n 2 `
    -y

Write-Host "`nDone. Results: results\terminal-bench\$JobName\result.json"
