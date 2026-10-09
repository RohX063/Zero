#!/usr/bin/env python3
"""Minimal UCI A/B match runner (stdlib only). Usage:
   match.py ENG_A ENG_B movetime_ms rounds [optA=val,...] [optB=val,...] [outfile]
Each opening is played twice with colours swapped. Reports A's score, W/D/L, Elo estimate.
"""
import subprocess, sys, re, math, time

SC = re.compile(r"score (cp|mate) (-?\d+)")
class Eng:
    def __init__(s, path, opts):
        s.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, bufsize=1)
        s.send("uci"); s.wait("uciok")
        for k, v in opts.items(): s.send(f"setoption name {k} value {v}")
        s.send("isready"); s.wait("readyok")
    def send(s, c): s.p.stdin.write(c + "\n"); s.p.stdin.flush()
    def wait(s, tok):
        while True:
            l = s.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            if l.startswith(tok): return
    def newgame(s): s.send("ucinewgame"); s.send("isready"); s.wait("readyok")
    def go(s, moves, mt):
        s.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        s.send(f"go movetime {mt}")
        score = None; best = "0000"
        while True:
            l = s.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            if l.startswith("info"):
                m = SC.search(l)
                if m:
                    v = int(m.group(2))
                    score = (100000 - abs(v) * 2 if v > 0 else -(100000 - abs(v) * 2)) if m.group(1) == "mate" and v != 0 else (-100000 if m.group(1) == "mate" else v)
                    if m.group(1) == "cp" and abs(v) >= 90000: score = v
            elif l.startswith("bestmove"):
                best = l.split()[1]; return best, score
    def close(s):
        try: s.send("quit"); s.p.wait(timeout=2)
        except Exception: s.p.kill()

def parse_opts(t):
    if not t or t == "-": return {}
    return {k: v for k, v in (x.split("=", 1) for x in t.split(","))}

def play(a, b, opening, mt, a_white, referee):
    white, black = (a, b) if a_white else (b, a)
    moves = list(opening); hist = []
    for ply in range(len(moves), 260):
        wtm = ply % 2 == 0
        eng = white if wtm else black
        best, score = eng.go(moves, mt)
        if best == "0000":
            # Terminal: ask the referee whether it is mate or stalemate.
            _, rs = referee.go(moves, 20)
            mated = rs is not None and rs <= -90000
            if mated:
                return (1.0 if (wtm != a_white) else 0.0)   # side to move was mated
            return 0.5
        moves.append(best)
        if score is not None:
            hist.append(score if wtm else -score)   # white-perspective
            h = hist[-4:]
            if len(h) == 4 and all(x >= 1200 for x in h): return 1.0 if a_white else 0.0
            if len(h) == 4 and all(x <= -1200 for x in h): return 0.0 if a_white else 1.0
            if ply > 100 and len(hist) >= 12 and all(abs(x) <= 8 for x in hist[-12:]): return 0.5
    return 0.5

def main():
    A, B, mt, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    oa = parse_opts(sys.argv[5] if len(sys.argv) > 5 else ""); ob = parse_opts(sys.argv[6] if len(sys.argv) > 6 else "")
    out = sys.argv[7] if len(sys.argv) > 7 else None
    ops = [l.split() for l in open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "openings24.txt")) if l.strip()]
    ea, eb, ref = Eng(A, oa), Eng(B, ob), Eng(A, {})
    w = d = l = 0; t0 = time.time()
    games = 0
    try:
        for r in range(rounds):
            op = ops[r % len(ops)]
            for a_white in (True, False):
                ea.newgame(); eb.newgame()
                s = play(ea, eb, op, mt, a_white, ref)
                games += 1
                if s == 1.0: w += 1
                elif s == 0.0: l += 1
                else: d += 1
                sc = (w + d / 2) / games
                msg = f"games {games} +{w} ={d} -{l} score {sc:.3f} t={time.time()-t0:.0f}s"
                if out: open(out, "w").write(msg + "\n")
    finally:
        for e in (ea, eb, ref): e.close()
    sc = (w + d / 2) / games
    sc = min(max(sc, 0.001), 0.999)
    elo = -400 * math.log10(1 / sc - 1)
    n = games; var = (w * (1 - sc) ** 2 + d * (0.5 - sc) ** 2 + l * sc ** 2) / n
    err = 400 / math.log(10) * math.sqrt(var / n) / (sc * (1 - sc)) * 1.96
    msg = f"FINAL A vs B: +{w} ={d} -{l}  score {sc:.3f}  Elo {elo:+.0f} +/- {err:.0f} (95%)  games {games}"
    print(msg)
    if out: open(out, "a").write(msg + "\n")
main()
