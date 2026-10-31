#!/usr/bin/env python3
"""ZERO SPSA tuner for the first logarithmic LMR parameter family.

Design goals:
- pure-stdlib: no python-chess dependency required;
- common-random-number paired self-play (+theta vs -theta);
- deterministic opening prefixes and color swapping;
- fixed-position regression guard against an anchor configuration;
- resumable state;
- node efficiency recorded as a secondary signal, never the sole objective.

This is a tuning harness, not a claim that any candidate is stronger until a
separate holdout and engine-match validation confirms it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


INFO_SCORE_RE = re.compile(r"\bscore\s+(cp\s+(-?\d+)|mate\s+(-?\d+))")
INFO_NODES_RE = re.compile(r"\bnodes\s+(\d+)")
INFO_NPS_RE = re.compile(r"\bnps\s+(\d+)")
INFO_DEPTH_RE = re.compile(r"\bdepth\s+(\d+)")


@dataclasses.dataclass
class Param:
    name: str
    value: int
    lo: int
    hi: int
    c0: float


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

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for name, value in self.options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok")

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

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def search(self, fen_or_start: str, moves: Sequence[str], movetime_ms: int, depth: int = 0) -> SearchReport:
        if fen_or_start == "startpos":
            position_cmd = "position startpos"
        else:
            position_cmd = f"position fen {fen_or_start}"
        if moves:
            position_cmd += " moves " + " ".join(moves)
        self._send(position_cmd)
        if depth > 0:
            self._send(f"go depth {depth}")
        else:
            self._send(f"go movetime {max(1, movetime_ms)}")

        bestmove = "0000"
        score_cp = 0.0
        nodes = 0
        nps = 0
        reported_depth = 0
        mate: Optional[int] = None
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("info "):
                score_match = INFO_SCORE_RE.search(line)
                if score_match:
                    if score_match.group(2) is not None:
                        score_cp = float(score_match.group(2))
                        mate = None
                    else:
                        mate = int(score_match.group(3))
                        score_cp = 32000.0 if mate > 0 else -32000.0
                nodes_match = INFO_NODES_RE.search(line)
                if nodes_match:
                    nodes = int(nodes_match.group(1))
                nps_match = INFO_NPS_RE.search(line)
                if nps_match:
                    nps = int(nps_match.group(1))
                depth_match = INFO_DEPTH_RE.search(line)
                if depth_match:
                    reported_depth = int(depth_match.group(1))
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                return SearchReport(bestmove, score_cp, nodes, nps, reported_depth, mate)
        raise TimeoutError("timeout waiting for bestmove")

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


def parse_params(path: Path) -> List[Param]:
    raw = json.loads(path.read_text())
    result: List[Param] = []
    for name in ("LMRBase", "LMRDepthCoeff", "LMRMoveCoeff"):
        item = raw[name]
        result.append(Param(name, int(item["value"]), int(item["min"]), int(item["max"]), float(item["ck"])))
    return result


def params_dict(params: Sequence[Param]) -> Dict[str, int]:
    return {p.name: int(p.value) for p in params}


def clipped_value(p: Param, value: float) -> int:
    return int(round(max(p.lo, min(p.hi, value))))


def perturb(params: Sequence[Param], signs: Sequence[int], ck_scale: float) -> List[int]:
    values = []
    for p, sign in zip(params, signs):
        values.append(clipped_value(p, p.value + sign * p.c0 * ck_scale))
    return values


def write_params(base: Sequence[Param], values: Sequence[int]) -> List[Param]:
    return [dataclasses.replace(p, value=int(v)) for p, v in zip(base, values)]


def load_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def load_fens(path: Path) -> List[str]:
    return load_lines(path)


def score_to_points(report: SearchReport) -> float:
    if report.bestmove != "0000":
        return 0.5  # not terminal; caller uses the final game state separately.
    if report.mate is None:
        return 0.5
    return 1.0 if report.mate > 0 else 0.0


def run_game(engine_path: str, white_options: Dict[str, int], black_options: Dict[str, int],
             opening: Sequence[str], movetime_ms: int, max_moves: int, game_seed: int) -> float:
    """Return WHITE score from the perspective of white_options."""
    del game_seed  # reserved for future opening randomization without nondeterminism in a run.
    white = UCIEngine(engine_path, white_options)
    black = UCIEngine(engine_path, black_options)
    moves: List[str] = list(opening)
    side_white = True if len(moves) % 2 == 0 else False
    try:
        white.start()
        black.start()
        white.new_game()
        black.new_game()
        for ply in range(max_moves):
            engine = white if side_white else black
            report = engine.search("startpos", moves, movetime_ms)
            if report.bestmove == "0000":
                # Score in ZERO is always from side-to-move perspective.
                if report.mate is None:
                    return 0.5
                side_score = 1.0 if report.mate > 0 else 0.0
                return side_score if side_white else 1.0 - side_score
            moves.append(report.bestmove)
            side_white = not side_white
        return 0.5
    finally:
        white.close()
        black.close()


def regression_guard(engine_path: str, candidate: Dict[str, int], anchor: Dict[str, int],
                     fens: Sequence[str], depth: int) -> Tuple[float, float, List[str]]:
    """Return (failure_rate, mean_nodes_ratio, failures).

    A failure is deliberately conservative: candidate changes the anchor's
    best move and also drops at least 80cp at the same fixed depth.
    """
    if not fens:
        return 0.0, 1.0, []

    failures: List[str] = []
    ratios: List[float] = []
    for fen in fens:
        with UCIEngine(engine_path, anchor) as a_eng, UCIEngine(engine_path, candidate) as c_eng:
            a = a_eng.search(fen, [], 0, depth=depth)
            c = c_eng.search(fen, [], 0, depth=depth)
        if a.nodes > 0 and c.nodes > 0:
            ratios.append(c.nodes / a.nodes)
        if c.bestmove != a.bestmove and c.score_cp <= a.score_cp - 80.0:
            failures.append(fen)

    return len(failures) / len(fens), (statistics.mean(ratios) if ratios else 1.0), failures


def objective(match_score: float, regression_failure_rate: float, node_ratio: float) -> float:
    # Match strength is primary. Regression damage receives a hard enough
    # penalty to discourage blind pruning. Small node-efficiency improvements
    # are useful, but cannot outweigh chess strength by themselves.
    efficiency = max(-1.0, min(1.0, 1.0 - node_ratio))
    return match_score - 0.75 * regression_failure_rate + 0.05 * efficiency


def main() -> int:
    ap = argparse.ArgumentParser(description="ZERO SPSA tuner for logarithmic LMR")
    ap.add_argument("--engine", required=True, help="Path to ZERO UCI binary")
    ap.add_argument("--param-file", default="tuning/lmr_params.json")
    ap.add_argument("--positions", default="tuning/regression.fen")
    ap.add_argument("--openings", default="tuning/openings.txt")
    ap.add_argument("--state", default="tuning/spsa_state.json")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--games", type=int, default=20, help="Games per SPSA direction pair; use even values")
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--regression-depth", type=int, default=7)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2601)
    ap.add_argument("--a0", type=float, default=32.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--a-exponent", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--a-offset", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    engine = os.path.abspath(args.engine)
    param_path = Path(args.param_file)
    openings_path = Path(args.openings)
    positions_path = Path(args.positions)
    state_path = Path(args.state)
    params = parse_params(param_path)
    anchor = params_dict(params)
    openings = [line.split() for line in load_lines(openings_path)]
    fens = load_fens(positions_path)
    if args.games % 2 != 0:
        raise SystemExit("--games must be even so candidate colors can be balanced.")

    if args.dry_run:
        rng = random.Random(args.seed)
        signs = [1 if rng.random() < 0.5 else -1 for _ in params]
        plus = perturb(params, signs, args.c_scale)
        minus = perturb(params, [-s for s in signs], args.c_scale)
        print(json.dumps({
            "anchor": anchor,
            "plus": dict(zip((p.name for p in params), plus)),
            "minus": dict(zip((p.name for p in params), minus)),
            "openings": len(openings),
            "regression_positions": len(fens),
        }, indent=2))
        return 0

    state = {"iteration": 0, "params": anchor, "history": []}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for p in params:
            p.value = int(state["params"][p.name])

    rng = random.Random(args.seed + int(state.get("iteration", 0)))
    start_iter = int(state.get("iteration", 0))

    for k in range(start_iter, start_iter + args.iterations):
        signs = [1 if rng.random() < 0.5 else -1 for _ in params]
        plus_values = perturb(params, signs, args.c_scale)
        minus_values = perturb(params, [-s for s in signs], args.c_scale)
        plus = write_params(params, plus_values)
        minus = write_params(params, minus_values)
        plus_cfg = params_dict(plus)
        minus_cfg = params_dict(minus)

        # Common-random-number schedule: same opening prefixes, same color order,
        # plus/minus swapped across pairs.
        match_points = 0.0
        total_games = 0
        game_indices = list(range(args.games // 2))
        rng.shuffle(game_indices)
        for i in game_indices:
            opening = openings[(k * 97 + i) % len(openings)] if openings else []
            # Candidate plus as White.
            result_w = run_game(engine, plus_cfg, minus_cfg, opening,
                                args.movetime, args.max_moves, game_seed=k * 1000 + i)
            # Candidate plus as Black.
            result_b = run_game(engine, minus_cfg, plus_cfg, opening,
                                args.movetime, args.max_moves, game_seed=k * 1000 + i + 500000)
            match_points += result_w
            match_points += 1.0 - result_b
            total_games += 2

        match_score = match_points / total_games if total_games else 0.5
        plus_reg = regression_guard(engine, plus_cfg, anchor, fens, args.regression_depth)
        minus_reg = regression_guard(engine, minus_cfg, anchor, fens, args.regression_depth)
        plus_obj = objective(match_score, plus_reg[0], plus_reg[1])
        # Mirror the same match score for minus by symmetry of the paired result.
        minus_match_score = 1.0 - match_score
        minus_obj = objective(minus_match_score, minus_reg[0], minus_reg[1])

        y = plus_obj - minus_obj
        ck = max(0.5, args.c_scale / ((k + 1) ** args.gamma))
        ak = args.a0 / ((k + args.a_offset) ** args.a_exponent)
        for idx, p in enumerate(params):
            grad = y / (2.0 * ck * signs[idx])
            p.value = clipped_value(p, p.value + ak * grad)

        state = {
            "iteration": k + 1,
            "params": params_dict(params),
            "history": state.get("history", []) + [{
                "iteration": k + 1,
                "plus": plus_cfg,
                "minus": minus_cfg,
                "match_score_plus": match_score,
                "plus_regression_failure_rate": plus_reg[0],
                "plus_node_ratio": plus_reg[1],
                "minus_regression_failure_rate": minus_reg[0],
                "minus_node_ratio": minus_reg[1],
                "plus_objective": plus_obj,
                "minus_objective": minus_obj,
                "y": y,
                "a_k": ak,
                "c_k": ck,
            }]
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))

        print(
            f"iter={k + 1} match={match_score:.3f} "
            f"reg+= {plus_reg[0]:.3f} reg-= {minus_reg[0]:.3f} "
            f"nodes+={plus_reg[1]:.3f} nodes-={minus_reg[1]:.3f} "
            f"params={params_dict(params)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
