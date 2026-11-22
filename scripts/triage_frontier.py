"""Bases needed to reach a coverage goal: triage (swept over D and tau) vs first-come vs random vs oracle.
goal: fraction of the genome covered at >= T x by the reads chosen for basecalling (T = 5 or 10), at 90 / 95 / 99 %.
usage: triage_frontier.py <online_rawsamble.paf> <truth_ref.paf>"""
import sys, importlib.util
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "3", "0.25"]
spec = importlib.util.spec_from_file_location("ts", __file__.replace("triage_frontier.py", "triage_sim.py"))
src = open(spec.origin).read().split('print(f"stream:')[0]       # reuse the stream/truth loading and policies, skip the report
ns = {"__file__": spec.origin}
exec(compile(src, spec.origin, "exec"), ns)
names, N, bases, aln, reflen = ns["names"], ns["N"], ns["bases"], ns["aln"], ns["reflen"]
run_triage, run_oracle, coverage = ns["run_triage"], ns["run_oracle"], ns["coverage"]
rng = np.random.default_rng(0)
tot = bases.sum()

def point(sel, T):
    cov = coverage(sel); return 100 * bases[list(sel)].sum() / tot, 100 * (cov >= T).mean()

def need(points, level):
    ok = [b for b, f in points if f >= level]
    return min(ok) if ok else float("nan")

def prefix_points(order_idx, T, steps=60):
    pts = []; cum = np.cumsum(bases[order_idx])
    for frac in np.linspace(0.05, 1.0, steps):
        k = int(np.searchsorted(cum, frac * tot)) + 1
        pts.append(point(list(order_idx[:k]), T))
    return pts

for T in (5, 10):
    fifo = prefix_points(np.arange(N), T)
    rnd = prefix_points(rng.permutation(N), T)
    tri, orc = [], []
    for D in range(2, 13):
        for tau in (0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            tri.append(point(run_triage(D, tau), T)); orc.append(point(run_oracle(D, tau), T))
    print(f"\n=== goal: fraction of genome covered at >= {T}x by the basecalled reads.  Basecalled bases (% of all) needed:")
    print(f"{'goal':>8} | {'TRIAGE (reference-free overlaps)':>32} | {'first-come':>10} | {'random':>7} | {'oracle (exact depth)':>20} | saving vs first-come")
    for level in (90, 95, 99):
        t, f_, r, o = need(tri, level), need(fifo, level), need(rnd, level), need(orc, level)
        print(f"{level:>7}% | {t:>32.1f} | {f_:>10.1f} | {r:>7.1f} | {o:>20.1f} | {100*(1-t/f_):>5.0f}%")
