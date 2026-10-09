# ZERO Phase 1 LMR SPSA — Windows Launch

PowerShell execution policy is intentionally not required.

## Build/verify once

From the project root:

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config RelWithDebInfo -j 4
ctest --test-dir build -C RelWithDebInfo --output-on-failure
```

## Start the overnight run

Double-click:

```text
scripts\run_phase1_overnight.cmd
```

or from a terminal:

```powershell
cmd /c .\scripts\run_phase1_overnight.cmd
```

This launcher does **not** execute a `.ps1` file. It invokes Python directly, runs CTest,
starts the 1000-iteration SPSA tuner, and writes a timestamped log under `tuning\logs`.

Do not use `run_phase1_overnight.ps1` for this experiment on machines where PowerShell
execution policies block local scripts.
