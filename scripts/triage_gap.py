"""R10 follow-up: (1) length-only policies vs triage, (2) why the oracle saves more: step-by-step decomposition.
Policies at equal goal (bases needed to cover >=T x on 90/95/99% of the genome):
  first-come, random, longest-first (offline, sees whole run), shortest-first,
  TRIAGE real (Rawsamble overlaps), ORACLE-SAME (exact reference overlaps but SAME strand only, depth D = what Rawsamble could see at best),
  ORACLE-BOTH (exact depth, both strands, 2D wanted = earlier 'oracle').
usage: triage_gap.py <online.paf> <truth_ref.paf>"""
import sys, importlib.util
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "3", "0.25"]
src = open(__file__.replace("triage_gap.py", "triage_sim.py")).read().split('print(f"stream:')[0]
ns = {"__file__": __file__}
exec(compile(src, "ts", "exec"), ns)
names, N, bases, aln, reflen = ns["names"], ns["N"], ns["bases"], ns["aln"], ns["reflen"]
run_triage, run_oracle, coverage = ns["run_triage"], ns["run_oracle"], ns["coverage"]
rng = np.random.default_rng(0); tot = bases.sum()

def run_oracle_same(D, tau):
    run = {'+': np.zeros(reflen, np.int32), '-': np.zeros(reflen, np.int32)}; sel = []
    for i, r in enumerate(names):
        a = aln.get(r)
        if not a: continue
        s, e, st = a; seg = run[st][s:e]
        if len(seg) and np.maximum(0, D - seg).mean() / D >= tau:
            run[st][s:e] += 1; sel.append(i)
    return sel

def point(sel, T):
    cov = coverage(sel); return 100 * bases[list(sel)].sum() / tot, 100 * (cov >= T).mean()
def need(pts, lv):
    ok = [b for b, f in pts if f >= lv]; return min(ok) if ok else float("nan")
def prefix(order_idx, T, steps=80):
    cum = np.cumsum(bases[order_idx]); out = []
    for fr in np.linspace(0.05, 1.0, steps):
        k = int(np.searchsorted(cum, fr * tot)) + 1; out.append(point(list(order_idx[:k]), T))
    return out
mapped = np.array([r in aln for r in names])
print(f"reads {N}; with alignment {mapped.sum()}; mean read {bases[bases>0].mean():.0f} b; median {np.median(bases[bases>0]):.0f} b; max {bases.max()}")
for T in (5, 10):
    P = {"first-come": prefix(np.arange(N), T), "random": prefix(rng.permutation(N), T),
         "longest-first (offline)": prefix(np.argsort(-bases), T), "shortest-first": prefix(np.argsort(bases), T)}
    tri, os_, ob = [], [], []
    for D in range(2, 13):
        for tau in (0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            tri.append(point(run_triage(D, tau), T)); os_.append(point(run_oracle_same(D, tau), T)); ob.append(point(run_oracle(D, tau), T))
    P["TRIAGE (real overlaps)"] = tri; P["ORACLE same-strand"] = os_; P["ORACLE both strands"] = ob
    print(f"\n=== goal: >= {T}x depth on X% of genome; bases needed (% of all bases)")
    print(f"{'policy':<26}" + "".join(f"{l:>8}%" for l in (90, 95, 99)))
    for k, v in P.items():
        print(f"{k:<26}" + "".join(f"{need(v, l):>9.1f}" for l in (90, 95, 99)))
