$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$BuildExe = Join-Path $Root 'build\RelWithDebInfo\zero.exe'
if (!(Test-Path $BuildExe)) {
    throw "ZERO executable not found: $BuildExe. Build the project first."
}

Write-Host '== ZERO Phase 1 LMR SPSA =='
Write-Host "Root: $Root"
Write-Host "Engine: $BuildExe"

ctest --test-dir build -C RelWithDebInfo --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "CTest failed with exit code $LASTEXITCODE" }

$logDir = Join-Path $Root 'tuning\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "phase1_$stamp.log"

$argsList = @(
    'tuning\spsa_lmr_phase1.py',
    '--engine', $BuildExe,
    '--param-file', 'tuning\lmr_phase1_params.json',
    '--state', 'tuning\spsa_lmr_phase1_state.json',
    '--best-file', 'tuning\lmr_phase1_best.json',
    '--openings', 'tuning\openings.txt',
    '--regression', 'tuning\regression.fen',
    '--holdout', 'tuning\holdout.fen',
    '--iterations', '1000',
    '--games', '10',
    '--jobs', '2',
    '--movetime', '50',
    '--max-plies', '300',
    '--adjudication-depth', '8',
    '--regression-depth', '8',
    '--checkpoint-every', '1'
)

Write-Host "Log: $logFile"
Write-Host 'Starting Phase-1 LMR SPSA run to iteration 1000...'

$ErrorActionPreference = 'Continue'
& python -u @argsList *>&1 | Tee-Object -FilePath $logFile
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "Python exited with code $exitCode" -ForegroundColor Red
    Write-Host "Full log: $logFile" -ForegroundColor Yellow
    exit $exitCode
}

Write-Host ''
Write-Host '== SPSA FINISHED =='
Write-Host 'Best candidate:'
Get-Content 'tuning\lmr_phase1_best.json'
