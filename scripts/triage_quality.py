"""Combined pre-basecall classifier: coverage need (Rawsamble overlaps with already-chosen reads, causal, reference-free) AND predicted read quality
(boosted trees on pre-basecall signal/overlap features, 2-fold cross-fitted so no read is predicted by a model that saw it).
Rule for each arriving read:  NOW if deficit >= tau  AND  predicted-quality percentile >= Q   (Q=0 is plain triage), else BACK BURNER.
Scoring (reference only for scoring): bases chosen (% of all), coverage of the genome by the chosen reads at >=T x, error of the chosen reads (1 - identity, base-weighted).
usage: triage_quality.py <online.paf> <coverage_truth.paf> <identity_truth.paf(-c)> <sigfeat.tsv> ; env CONTIG, TRI_W"""
import sys, os, importlib.util
import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import rankdata
paf, cov_paf, id_paf, sigf = sys.argv[1:5]
sys.argv = [sys.argv[0], paf, cov_paf, "3", "0.25"]
src = open(__file__.replace("triage_quality.py", "triage_sim.py")).read().split('print(f"stream:')[0]
ns = {"__file__": __file__}; exec(compile(src, "ts", "exec"), ns)
names, N, bases, aln, reflen, recs, order, qlen, W = (ns[k] for k in ("names", "N", "bases", "aln", "reflen", "recs", "order", "qlen", "W"))
CONTIG = os.environ.get("CONTIG")
ident = {}
for l in open(id_paf):
    c = l.split("\t")
    if c[0] in ident or (CONTIG and c[5] != CONTIG) or int(c[11]) < 20: continue
    ident[c[0]] = int(c[9]) / int(c[10])
sf = {}
for l in list(open(sigf))[1:]:
    c = l.split("\t"); sf[c[0]] = [float(x) for x in c[1:]]
ovl = defaultdict(lambda: [0, 0, 0.0, 0])
for l in open(paf):
    c = l.rstrip("\n").split("\t")
    if c[5] == "*": continue
    s1 = int([t for t in c[12:] if t.startswith("s1:i:")][0][5:]); o = ovl[c[0]]; o[0] += 1; o[1] = max(o[1], s1); o[2] += s1; o[3] += int(c[3]) - int(c[2])
def feat(r): o = ovl[r]; return sf[r] + [qlen[r], o[0], o[1], (o[2] / o[0]) if o[0] else 0.0, o[3] / qlen[r]]
have = np.array([r in sf and r in qlen for r in names]); X = np.zeros((N, 10)); X[have] = [feat(r) for r, h in zip(names, have) if h]
y = np.array([ident.get(r, np.nan) for r in names]); lab = have & np.isfinite(y)
half = N // 2; fold = np.arange(N) >= half; pred = np.full(N, np.nan)
for f in (0, 1):
    tr = lab & (fold != bool(f)); te = have & (fold == bool(f))
    pred[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(X[tr], y[tr]).predict(X[te])
pred[~have] = np.nanmedian(pred)
qpct = (rankdata(pred) - 1) / (N - 1)                       # predicted-quality percentile of each read within the run
errw = np.where(np.isfinite(y), 1 - y, np.nanmean(1 - y)); tot = bases.sum()
print(f"stream {N} reads; predictor out-of-fold Spearman with true identity {np.corrcoef(rankdata(pred[lab]), rankdata(y[lab]))[0,1]:+.3f}")

def run(D, tau, Q):
    chosen = np.zeros(N, bool); sel = []
    for i, r in enumerate(names):
        if qpct[i] < Q: continue
        nW = max(1, -(-qlen[r] // W)); diff = np.zeros(nW + 1, np.int32); by_t = defaultdict(list)
        for t, qs, qe in recs[r]:
            if chosen[order[t]]: by_t[t].append((qs, qe))
        for t, iv in by_t.items():
            iv.sort(); cs, ce = iv[0]
            for s, e in iv[1:] + [(None, None)]:
                if s is not None and s <= ce: ce = max(ce, e); continue
                lo, hi = cs // W, min(nW, -(-ce // W))
                if hi > lo: diff[lo] += 1; diff[hi] -= 1
                if s is not None: cs, ce = s, e
        if np.maximum(0, D - np.cumsum(diff)[:nW]).mean() / D >= tau: chosen[i] = True; sel.append(i)
    return sel

def score(sel, Ts=(1, 2, 5)):
    cov = ns["coverage"](sel); b = bases[sel]
    return 100 * b.sum() / tot, [100 * (cov >= T).mean() for T in Ts], 100 * np.average(errw[sel], weights=np.maximum(b, 1))
res = []
for Q in (0.0, 0.3, 0.5, 0.7):
    for D in (1, 2, 3, 5, 8):
        for tau in (0.1, 0.25, 0.5, 0.75):
            sel = run(D, tau, Q); b, cv, er = score(sel); res.append((Q, D, tau, b, cv, er))
print(f"all-reads error {100*np.average(errw, weights=np.maximum(bases,1)):.2f}%  (base-weighted)")
for T_i, T in enumerate((1, 2, 5)):
    print(f"\n=== goal: genome covered at >= {T}x by chosen reads.  best (fewest bases) setting reaching the goal, per quality gate Q  [Q=0 is triage without quality]")
    for lv in (90, 95):
        print(f"  goal {lv}%:  " + "  |  ".join(
            (lambda ok: f"Q={Q}: {min(ok, key=lambda r: r[3])[3]:5.1f}% bases, error {min(ok, key=lambda r: r[3])[5]:5.2f}%" if ok else f"Q={Q}: not reached")(
                [r for r in res if r[0] == Q and r[4][T_i] >= lv]) for Q in (0.0, 0.3, 0.5, 0.7)))

# ---- baselines at the same goals: no overlap logic.  GATE ONLY = random reads among those with predicted quality >= Q (Q=0: plain random); FIRST-COME = stream prefix.
rng = np.random.default_rng(0)
def prefix_best(idx_order, T_i, lv):
    cum = np.cumsum(bases[idx_order]); best = None
    for frac in np.linspace(0.02, 1.0, 120):
        k = int(np.searchsorted(cum, frac * tot)) + 1
        if k > len(idx_order): break
        b, cv, er = score(list(idx_order[:k]))
        if cv[T_i] >= lv: return b, er
    return None
print("\n=== BASELINES (no overlap logic): same goals.  'gate only' = random reads among those with predicted quality >= Q;  first-come = stream prefix")
for T_i, T in enumerate((1, 2, 5)):
    for lv in (90, 95):
        out = []
        for Q in (0.0, 0.3, 0.5, 0.7):
            el = np.where(qpct >= Q)[0]; r_ = prefix_best(rng.permutation(el), T_i, lv)
            out.append(f"gate Q={Q}: " + (f"{r_[0]:5.1f}% bases, error {r_[1]:5.2f}%" if r_ else "not reached"))
        fc = prefix_best(np.arange(N), T_i, lv)
        print(f"  >= {T}x on {lv}%:  " + "  |  ".join(out) + "  |  first-come: " + (f"{fc[0]:5.1f}%, {fc[1]:5.2f}%" if fc else "not reached"))
