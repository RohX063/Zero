#!/usr/bin/env python3
"""ZERO Phase-1 SPSA tuner for contextual logarithmic LMR.

Ten LMR parameters are tuned together while NMP, evaluation, TT policy,
move generation and the frozen depth*move interaction remain fixed.

This is an overnight experiment harness. It is deliberately conservative:
- paired +/- SPSA matches with common opening/color schedule;
- fixed regression gate against the phase baseline;
- separate holdout file is never used during optimization;
- resumable JSON state and best-candidate snapshots;
- optional small parallel game batches for 4-core-class machines;
- node ratio is reported but cannot dominate chess strength;
- deterministic RNG and opening schedule for reproducibility.

A candidate is NOT accepted as an Elo gain until an independent holdout and
external gauntlet confirm it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCORE_RE = re.compile(r"\bscore\s+(cp\s+(-?\d+)|mate\s+(-?\d+))")
NODES_RE = re.compile(r"\bnodes\s+(\d+)")
NPS_RE = re.compile(r"\bnps\s+(\d+)")
DEPTH_RE = re.compile(r"\bdepth\s+(\d+)")
PARAM_NAMES = (
    "LMRBase", "LMRDepthCoeff", "LMRMoveCoeff", "LMRPVAdjust",
    "LMRCutAdjust", "LMRTTAdjust", "LMRHistoryCoeff",
    "LMRContinuationCoeff", "LMRImprovingAdjust", "LMRTacticalSafetyAdjust",
)

@dataclasses.dataclass
class Param:
    name: str
    value: int
    lo: int
    hi: int
    ck: float
    ak: float

@dataclasses.dataclass
class SearchReport:
    bestmove: str
    score_cp: float
    nodes: int
    nps: int
    depth: int
    mate: Optional[int] = None

class UCIEngine:
    def __init__(self, path: str, options: Dict[str, int], timeout: float = 30.0):
        self.path = path
        self.options = options
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen[str]] = None

    def _send(self, command: str) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("UCI process is not running")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("UCI process is not running")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("UCI engine exited unexpectedly")
        return line.rstrip("\n")

    def _wait_for(self, token: str) -> None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if token in self._readline():
                return
        raise TimeoutError(f"timeout waiting for {token}")

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for name, value in self.options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def search(self, moves: Sequence[str], movetime_ms: int) -> SearchReport:
        cmd = "position startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)
        self._send(f"go movetime {max(1, movetime_ms)}")
        bestmove = "0000"
        score_cp = 0.0
        nodes = 0
        nps = 0
        depth = 0
        mate = None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("info "):
                m = SCORE_RE.search(line)
                if m:
                    if m.group(2) is not None:
                        score_cp = float(m.group(2))
                        mate = None
                    else:
                        mate = int(m.group(3))
                        score_cp = 32000.0 if mate > 0 else -32000.0
                m = NODES_RE.search(line)
                if m: nodes = int(m.group(1))
                m = NPS_RE.search(line)
                if m: nps = int(m.group(1))
                m = DEPTH_RE.search(line)
                if m: depth = int(m.group(1))
            elif line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                return SearchReport(bestmove, score_cp, nodes, nps, depth, mate)
        raise TimeoutError("timeout waiting for bestmove")

    def search_fen(self, fen: str, depth: int) -> SearchReport:
        self._send(f"position fen {fen}")
        self._send(f"go depth {max(1, depth)}")
        last = SearchReport("0000", 0.0, 0, 0, 0, None)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("info "):
                m = SCORE_RE.search(line)
                if m:
                    if m.group(2) is not None:
                        last.score_cp = float(m.group(2)); last.mate = None
                    else:
                        last.mate = int(m.group(3)); last.score_cp = 32000.0 if last.mate > 0 else -32000.0
                m = NODES_RE.search(line)
                if m: last.nodes = int(m.group(1))
                m = DEPTH_RE.search(line)
                if m: last.depth = int(m.group(1))
            elif line.startswith("bestmove "):
                last.bestmove = line.split()[1]
                return last
        raise TimeoutError("timeout waiting for FEN bestmove")

    def evaluate_position(self, moves: Sequence[str], depth: int) -> SearchReport:
        """Evaluate the current startpos+move-list position at fixed depth."""
        cmd = "position startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)
        self._send(f"go depth {max(1, depth)}")
        last = SearchReport("0000", 0.0, 0, 0, 0, None)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("info "):
                m = SCORE_RE.search(line)
                if m:
                    if m.group(2) is not None:
                        last.score_cp = float(m.group(2)); last.mate = None
                    else:
                        last.mate = int(m.group(3)); last.score_cp = 32000.0 if last.mate > 0 else -32000.0
                m = NODES_RE.search(line)
                if m: last.nodes = int(m.group(1))
                m = DEPTH_RE.search(line)
                if m: last.depth = int(m.group(1))
            elif line.startswith("bestmove "):
                parts = line.split()
                if len(parts) >= 2:
                    last.bestmove = parts[1]
                return last
        raise TimeoutError("timeout waiting for adjudication search")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self._send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self) -> "UCIEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_params(path: Path) -> List[Param]:
    raw = json.loads(path.read_text())
    return [Param(name, int(raw[name]["value"]), int(raw[name]["min"]),
                  int(raw[name]["max"]), float(raw[name]["ck"]), float(raw[name]["ak"]))
            for name in PARAM_NAMES]


def save_params(path: Path, params: Sequence[Param]) -> None:
    raw = json.loads(path.read_text())
    for p in params:
        raw[p.name]["value"] = int(p.value)
    path.write_text(json.dumps(raw, indent=2) + "\n")


def cfg(params: Sequence[Param]) -> Dict[str, int]:
    return {p.name: int(p.value) for p in params}


def clamp(p: Param, x: float) -> int:
    return int(round(max(p.lo, min(p.hi, x))))


def perturb(params: Sequence[Param], signs: Sequence[int], ck_mult: float) -> List[int]:
    return [clamp(p, p.value + s * p.ck * ck_mult) for p, s in zip(params, signs)]


def load_noncomment_lines(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")]


def outcome(report: SearchReport, side_is_white: bool) -> float:
    if report.bestmove != "0000":
        return 0.5
    if report.mate is None:
        return 0.5
    side_score = 1.0 if report.mate > 0 else 0.0
    return side_score if side_is_white else 1.0 - side_score


def terminal_result(report: SearchReport, side_is_white: bool) -> Optional[float]:
    """Return a terminal result from the '+' engine perspective.

    report.mate is from the side-to-move perspective. A positive mate score
    means the side to move can force mate; a negative score means the side to
    move is getting mated.
    """
    if report.bestmove != "0000":
        return None
    if report.mate is None:
        return 0.5  # no legal move without mate => stalemate/draw
    side_wins = report.mate > 0
    white_wins = side_wins if side_is_white else not side_wins
    return 1.0 if white_wins else 0.0


def soft_result_from_score(score_cp: float, plus_is_side_to_move: bool) -> float:
    """Bounded fallback score for fast games that hit the ply cap.

    This is a search-tuning signal, not an Elo/result claim. Exact terminal
    results always take precedence.
    """
    cp = score_cp if plus_is_side_to_move else -score_cp
    cp = max(-1000.0, min(1000.0, cp))
    return 0.5 + 0.5 * math.tanh(cp / 500.0)


def play_batch(engine_path: str, plus_cfg: Dict[str, int], minus_cfg: Dict[str, int],
               pairs: Sequence[Tuple[Sequence[str], int]], movetime: int, max_plies: int,
               adjudication_depth: int) -> List[float]:
    """Run paired games while keeping UCI processes alive.

    Exact mate/stalemate results are used when a game terminates. For fast
    experiments that reach max_plies, a bounded score-based adjudication is
    used only to keep the SPSA objective from becoming a flat all-draw signal.
    """
    white = UCIEngine(engine_path, plus_cfg)
    black = UCIEngine(engine_path, minus_cfg)
    results: List[float] = []
    try:
        white.start(); black.start()
        for opening, seed in pairs:
            del seed
            for plus_is_black in (False, True):
                moves = list(opening)
                side_white = (len(moves) % 2 == 0)
                white.new_game(); black.new_game()

                # Which persistent engine represents the '+' candidate in this
                # leg of the color-swapped pair?
                plus_engine = black if plus_is_black else white

                for _ in range(max_plies):
                    engine = black if (side_white == plus_is_black) else white
                    report = engine.search(moves, movetime)
                    terminal = terminal_result(report, side_white)
                    if terminal is not None:
                        results.append(1.0 - terminal if plus_is_black else terminal)
                        break
                    moves.append(report.bestmove)
                    side_white = not side_white
                else:
                    # Final bounded adjudication for unfinished fast games.
                    current_engine = black if plus_is_black == side_white else white
                    final = current_engine.evaluate_position(moves, adjudication_depth)
                    results.append(
                        soft_result_from_score(
                            final.score_cp,
                            plus_is_side_to_move=(current_engine is plus_engine),
                        )
                    )
    finally:
        white.close(); black.close()
    return results



def regression_guard(engine_path: str, candidate: Dict[str, int], anchor: Dict[str, int],
                     fens: Sequence[str], depth: int) -> Tuple[float, float, List[str]]:
    failures: List[str] = []
    ratios: List[float] = []
    if not fens:
        return 0.0, 1.0, failures
    with UCIEngine(engine_path, anchor) as a_eng, UCIEngine(engine_path, candidate) as c_eng:
        a_eng.new_game(); c_eng.new_game()
        for fen in fens:
            a = a_eng.search_fen(fen, depth)
            c = c_eng.search_fen(fen, depth)
            if a.nodes and c.nodes:
                ratios.append(c.nodes / a.nodes)
            if c.bestmove != a.bestmove and c.score_cp <= a.score_cp - 80.0:
                failures.append(fen)
    return (len(failures) / len(fens), statistics.mean(ratios) if ratios else 1.0, failures)



def objective(match_score: float, reg_failure_rate: float, node_ratio: float) -> float:
    # Match score dominates. Regression damage is a hard penalty. Node efficiency
    # is a small tiebreaker, not the main target.
    efficiency = max(-1.0, min(1.0, 1.0 - node_ratio))
    return match_score - 1.50 * reg_failure_rate + 0.03 * efficiency


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--param-file", default="tuning/lmr_phase1_params.json")
    ap.add_argument("--state", default="tuning/spsa_lmr_phase1_state.json")
    ap.add_argument("--best-file", default="tuning/lmr_phase1_best.json")
    ap.add_argument("--openings", default="tuning/openings.txt")
    ap.add_argument("--regression", default="tuning/regression.fen")
    ap.add_argument("--holdout", default="tuning/holdout.fen")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--games", type=int, default=10, help="games per iteration; must be even")
    ap.add_argument("--jobs", type=int, default=2, help="parallel paired games; use 1-2 on a 4-core machine")
    ap.add_argument("--movetime", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--adjudication-depth", type=int, default=8)
    ap.add_argument("--regression-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=260104)
    ap.add_argument("--a-decay", type=float, default=0.602)
    ap.add_argument("--c-decay", type=float, default=0.101)
    ap.add_argument("--a-offset", type=float, default=50.0)
    ap.add_argument("--c-mult", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.games < 2 or args.games % 2:
        raise SystemExit("--games must be even and >= 2")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    engine = os.path.abspath(args.engine)
    param_path = Path(args.param_file)
    state_path = Path(args.state)
    best_path = Path(args.best_file)
    openings = [x.split() for x in load_noncomment_lines(Path(args.openings))]
    regression = load_noncomment_lines(Path(args.regression))
    holdout = load_noncomment_lines(Path(args.holdout))
    params = load_params(param_path)
    anchor = {p.name: p.value for p in params}

    if args.dry_run:
        rng = random.Random(args.seed)
        signs = [1 if rng.random() < 0.5 else -1 for _ in params]
        plus = perturb(params, signs, 1.0)
        minus = perturb(params, [-s for s in signs], 1.0)
        print(json.dumps({
            "parameters": PARAM_NAMES,
            "anchor": anchor,
            "plus": dict(zip(PARAM_NAMES, plus)),
            "minus": dict(zip(PARAM_NAMES, minus)),
            "iterations": args.iterations,
            "games_per_iteration": args.games,
            "jobs": args.jobs,
            "holdout_positions_loaded": len(holdout),
        }, indent=2))
        return 0

    if state_path.exists():
        state = json.loads(state_path.read_text())
        for p in params:
            p.value = int(state["params"][p.name])
        start = int(state.get("iteration", 0))
        best = state.get("best", {"params": {p.name: p.value for p in params}, "objective": -1e9})
        history = list(state.get("history", []))
        rng = random.Random(int(state.get("rng_state_seed", args.seed)) + start)
    else:
        start = 0
        best = {"params": {p.name: p.value for p in params}, "objective": -1e9}
        history = []
        rng = random.Random(args.seed)

    end = args.iterations
    if start >= end:
        print(f"Phase 1 already complete through iteration {start}; target={end}")
        return 0
    print(f"Phase 1 LMR SPSA: iterations {start + 1}..{end}, games/iter={args.games}, jobs={args.jobs}")
    print(f"Regression positions={len(regression)}, holdout positions loaded={len(holdout)} (HOLDOUT UNUSED)")

    for k in range(start, end):
        signs = [1 if rng.random() < 0.5 else -1 for _ in params]
        plus_values = perturb(params, signs, args.c_mult)
        minus_values = perturb(params, [-s for s in signs], args.c_mult)
        plus_cfg = dict(zip(PARAM_NAMES, plus_values))
        minus_cfg = dict(zip(PARAM_NAMES, minus_values))

        # Common-random-number paired schedule. Each worker keeps its two engine
        # processes alive for several pairs; this removes thousands of costly
        # process startups during an overnight run.
        pair_specs: List[Tuple[Sequence[str], int]] = []
        for i in range(args.games // 2):
            opening = openings[(k * 131 + i) % len(openings)] if openings else []
            pair_specs.append((opening, k * 1000 + i))

        chunks: List[List[Tuple[Sequence[str], int]]] = [[] for _ in range(min(args.jobs, len(pair_specs)))]
        for i, spec in enumerate(pair_specs):
            chunks[i % len(chunks)].append(spec)

        pair_results: List[float] = []
        if len(chunks) == 1:
            pair_results.extend(play_batch(engine, plus_cfg, minus_cfg, chunks[0], args.movetime, args.max_plies, args.adjudication_depth))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                futures = [ex.submit(play_batch, engine, plus_cfg, minus_cfg, chunk, args.movetime, args.max_plies, args.adjudication_depth)
                           for chunk in chunks if chunk]
                for fut in futures:
                    pair_results.extend(fut.result())

        match_score_plus = sum(pair_results) / args.games if pair_results else 0.5
        plus_reg = regression_guard(engine, plus_cfg, anchor, regression, args.regression_depth)
        minus_reg = regression_guard(engine, minus_cfg, anchor, regression, args.regression_depth)
        plus_obj = objective(match_score_plus, plus_reg[0], plus_reg[1])
        minus_match = 1.0 - match_score_plus
        minus_obj = objective(minus_match, minus_reg[0], minus_reg[1])
        y = plus_obj - minus_obj

        # Classical SPSA gain schedules with per-parameter step scales.
        ck = max(0.25, args.c_mult / ((k + 1.0) ** args.c_decay))
        ak_base = 1.0 / ((k + args.a_offset) ** args.a_decay)
        for idx, p in enumerate(params):
            if signs[idx] == 0:
                continue
            gradient = y / (2.0 * ck * signs[idx])
            p.value = clamp(p, p.value + p.ak * ak_base * gradient)

        current = {p.name: p.value for p in params}
        rec = {
            "iteration": k + 1,
            "plus": plus_cfg,
            "minus": minus_cfg,
            "match_score_plus": match_score_plus,
            "plus_regression_failure_rate": plus_reg[0],
            "minus_regression_failure_rate": minus_reg[0],
            "plus_node_ratio": plus_reg[1],
            "minus_node_ratio": minus_reg[1],
            "plus_objective": plus_obj,
            "minus_objective": minus_obj,
            "delta_objective": y,
            "ak_base": ak_base,
            "ck": ck,
            "params_after": current,
            "plus_regression_failures": plus_reg[2],
            "minus_regression_failures": minus_reg[2],
        }
        history.append(rec)

        if plus_obj > float(best.get("objective", -1e9)) and plus_reg[0] == 0.0:
            best = {"iteration": k + 1, "objective": plus_obj, "params": plus_cfg,
                    "match_score": match_score_plus, "node_ratio": plus_reg[1]}
            best_path.write_text(json.dumps(best, indent=2) + "\n")

        if (k + 1) % args.checkpoint_every == 0:
            state = {
                "iteration": k + 1,
                "params": current,
                "best": best,
                "history": history,
                "rng_state_seed": args.seed,
                "config": vars(args),
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            save_params(param_path, [dataclasses.replace(p) for p in params])

        print(
            f"iter={k+1:04d} score+={match_score_plus:.3f} "
            f"objΔ={y:+.4f} reg+={plus_reg[0]:.3f} reg-={minus_reg[0]:.3f} "
            f"nodes+={plus_reg[1]:.3f} nodes-={minus_reg[1]:.3f} "
            f"best={best.get('objective', -1e9):+.4f} "
            f"params={current}", flush=True
        )

    # Never touch holdout during tuning; record that it was intentionally unused.
    summary = {
        "completed_to_iteration": end,
        "best": best,
        "holdout_positions_reserved": len(holdout),
        "note": "Run an independent holdout/gauntlet after tuning; do not select by holdout results during this cycle.",
    }
    Path("tuning/spsa_phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PHASE1_COMPLETE")
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
