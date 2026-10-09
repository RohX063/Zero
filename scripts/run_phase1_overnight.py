#!/usr/bin/env python3
"""Windows-friendly ZERO Phase-1 LMR SPSA launcher.

This launcher avoids PowerShell execution-policy issues entirely. It runs CTest,
then the 1000-iteration SPSA tuner, while teeing output to a timestamped log.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    engine = root / "build" / "RelWithDebInfo" / "zero.exe"
    if not engine.exists():
        print(f"ERROR: ZERO executable not found: {engine}")
        print("Build first with:")
        print("  cmake --build build --config RelWithDebInfo -j 4")
        return 2

    print("== ZERO Phase 1 LMR SPSA ==")
    print(f"Root:   {root}")
    print(f"Engine: {engine}")

    ctest = shutil.which("ctest") or "ctest"
    print("Running CTest...")
    rc = subprocess.run(
        [ctest, "--test-dir", "build", "-C", "RelWithDebInfo", "--output-on-failure"],
        cwd=root,
    ).returncode
    if rc != 0:
        print(f"ERROR: CTest failed with exit code {rc}")
        return rc

    log_dir = root / "tuning" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase1_{stamp}.log"

    args = [
        sys.executable, "-u", "tuning/spsa_lmr_phase1.py",
        "--engine", str(engine),
        "--param-file", "tuning/lmr_phase1_params.json",
        "--state", "tuning/spsa_lmr_phase1_state.json",
        "--best-file", "tuning/lmr_phase1_best.json",
        "--openings", "tuning/openings.txt",
        "--regression", "tuning/regression.fen",
        "--holdout", "tuning/holdout.fen",
        "--iterations", "1000",
        "--games", "10",
        "--jobs", "2",
        "--movetime", "50",
        "--max-plies", "300",
        "--adjudication-depth", "8",
        "--regression-depth", "8",
        "--checkpoint-every", "1",
    ]

    print(f"Log: {log_file}")
    print("Starting 1000-iteration Phase-1 LMR SPSA run...")
    print("No PowerShell script is involved; this launcher is execution-policy independent.")

    with log_file.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            args,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = proc.wait()

    if rc != 0:
        print(f"Python tuner exited with code {rc}")
        print(f"Full log: {log_file}")
        return rc

    print("== SPSA FINISHED ==")
    best = root / "tuning" / "lmr_phase1_best.json"
    if best.exists():
        print("Best candidate:")
        print(best.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
