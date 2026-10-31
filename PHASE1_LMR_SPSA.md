# ZERO 2.5 R3.3 — Phase 1 LMR SPSA Overnight Run

This branch is the first full LMR-only SPSA experiment.

## Tuned parameters (10)

1. LMRBase
2. LMRDepthCoeff
3. LMRMoveCoeff
4. LMRPVAdjust
5. LMRCutAdjust
6. LMRTTAdjust
7. LMRHistoryCoeff
8. LMRContinuationCoeff
9. LMRImprovingAdjust
10. LMRTacticalSafetyAdjust

The base logarithmic depth×move interaction is frozen during Phase 1. NMP,
TT policy, evaluation, move generation, and the narrow-window adjustment remain
fixed so that an observed change can be attributed primarily to contextual LMR.

## Overnight configuration

- 1000 SPSA iterations
- 10 games per iteration (5 paired openings, both color assignments)
- 2 concurrent paired-game workers (appropriate for a 4-core-class machine)
- 50 ms per move
- 100 plies maximum per game
- regression gate at depth 8
- holdout positions are never used by SPSA
- checkpoint after every iteration

## Windows command

From the project root:

```powershell
.\scripts\run_phase1_overnight.ps1
```

Or directly:

```powershell
py tuning\spsa_lmr_phase1.py `
  --engine .\build\RelWithDebInfo\zero.exe `
  --iterations 1000 `
  --games 10 `
  --jobs 2 `
  --movetime 50 `
  --max-plies 100 `
  --regression-depth 8
```

The run is resumable. If interrupted, rerun the same command; the state file
continues from the last checkpoint.

## What to inspect afterward

- `tuning/spsa_lmr_phase1_state.json` — complete optimization history
- `tuning/lmr_phase1_best.json` — best candidate seen during tuning
- `tuning/spsa_phase1_summary.json` — final summary
- `tuning/logs/` — console log

Do not call the best candidate an Elo gain yet. First run the independent
holdout and an external matched gauntlet with fixed hardware/time control.
