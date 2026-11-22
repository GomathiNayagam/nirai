"""Independent-dataset test (K. pneumoniae KPC2, FLO-MIN114 R10.4.1, flow cell FAX76295) of (1) the pre-basecall noise->quality predictor FROZEN from the E. coli R10 run and
(2) overlap triage + quality gate.  Nothing here is fitted on KPC2 labels.
Inputs: online Rawsamble PAF of the stream, signal-to-reference truth PAF (coverage truth, positions extended to the full read), signal features of the stream,
identity of the randomly basecalled labeled subset (minimap2 of Dorado hac reads to the isolate reference).
usage: kpc2_triage_test.py <online.paf> <truth_sig.paf> <sigfeat.tsv> <labeled_hac.paf>   env TRI_W (default 1000)"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
paf, tpaf, sigf, lpaf = sys.argv[1:5]
os.environ.setdefault("CONTIG", "chromosome"); os.environ.setdefault("TRI_W", "1000")
F = ["n_samples", "sd", "mad_diff", "med_pa", "frac_hi"]
# ---- frozen predictor: trained ONLY on the E. coli R10 run (60k reads), signal features only
R = "/home/sgomathi/rawsamble_work/r10_run/"
tr = pd.read_csv(R + "sigfeat_60k.tsv", sep="\t"); idn = {}
for l in open(R + "truth_c_60k.paf"):
    c = l.split("\t")
    if c[0] not in idn and int(c[11]) >= 20: idn[c[0]] = int(c[9]) / int(c[10])
tr = tr[tr.read.isin(idn)]; model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[F], tr.read.map(idn))
print(f"predictor trained on {len(tr)} E. coli R10 reads (signal features only), frozen")
# ---- reuse the triage machinery (stream, overlaps, coverage truth)
sys.argv = [sys.argv[0], paf, tpaf, "3", "0.25"]
src = open(__file__.replace("kpc2_triage_test.py", "triage_sim.py")).read().split('print(f"stream:')[0]
ns = {"__file__": __file__}; exec(compile(src, "ts", "exec"), ns)
names, N, bases, aln, reflen, recs, order, qlen = (ns[k] for k in ("names", "N", "bases", "aln", "reflen", "recs", "order", "qlen")); W = ns["W"]
sf = pd.read_csv(sigf, sep="\t").set_index("read").reindex(names)
pred = model.predict(sf[F].fillna(sf[F].median())); qpct = (rankdata(pred) - 1) / (N - 1); noise = sf.mad_diff.values
lab = {}
for l in open(lpaf):
    c = l.split("\t")
    if c[0] not in lab and int(c[11]) >= 20: lab[c[0]] = int(c[9]) / int(c[10])
y = np.array([lab.get(r, np.nan) for r in names]); L = np.isfinite(y)
print(f"stream {N} reads, genome (chromosome) {reflen/1e6:.2f} Mb, {bases.sum()/1e6:.0f} Mb estimated bases; labeled+aligned reads: {L.sum()}")
# ---- Part 1: predictor transfer on labeled reads
yl, bl = y[L], bases[L]
err = lambda m: 100 * (1 - np.average(y[m], weights=np.maximum(bases[m], 1)))
rng = np.random.default_rng(0)
print("\n=== PART 1: does noise predict identity on this independent run?  (labeled reads, n=%d, all-reads error %.2f%%)" % (L.sum(), err(L)))
print("Spearman: noise(mad_diff) vs identity %+.3f | frozen R10 predictor vs identity %+.3f | read length vs identity %+.3f | noise vs length %+.3f" % (
    spearmanr(noise[L], yl)[0], spearmanr(pred[L], yl)[0], spearmanr(bases[L], yl)[0], spearmanr(noise[L], bl)[0]))
idx = np.where(L)[0]; k = len(idx) // 4
for nm, key in (("quietest 25% of reads", noise[idx]), ("predictor top 25%", -pred[idx]), ("longest 25%", -bases[idx])):
    ch = idx[np.argsort(key)[:k]]; m = np.zeros(N, bool); m[ch] = True
    print(f"  {nm:<24} error {err(m):5.2f}%   share of labeled bases {100*bases[ch].sum()/bases[idx].sum():5.1f}%")
rr = [err(np.isin(np.arange(N), rng.choice(idx, k, replace=False))) for _ in range(200)]
print(f"  random 25% of reads      error {np.mean(rr):5.2f}%  (sd {np.std(rr):.2f})")
bs = []
for _ in range(300):
    s = rng.choice(idx, len(idx)); bs.append(spearmanr(noise[s], y[s])[0])
print("  bootstrap 95%% CI of Spearman(noise, identity): [%+.3f, %+.3f]" % tuple(np.percentile(bs, [2.5, 97.5])))
# ---- Part 2: triage + quality gate
errw = lambda sel: (100 * (1 - np.average(y[sel][np.isfinite(y[sel])], weights=np.maximum(bases[sel][np.isfinite(y[sel])], 1))) if np.isfinite(y[sel]).sum() > 20 else np.nan)
tot = bases.sum()
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
    cov = ns["coverage"](sel); b = bases[sel]; return 100 * b.sum() / tot, [100 * (cov >= T).mean() for T in Ts], errw(np.array(sel))
res = []
for Q in (0.0, 0.3, 0.5, 0.7):
    for D in (1, 2, 3, 5):
        for tau in (0.1, 0.25, 0.5, 0.75):
            sel = run(D, tau, Q); b, cv, er = score(sel); res.append((Q, D, tau, b, cv, er))
print(f"\n=== PART 2: overlap triage + quality gate.  all-reads error {errw(np.arange(N)):.2f}%.  Coverage truth = signal-to-reference mapping (only reads mapping with MAPQ>=20 count; same for all policies)")
full = score(list(range(N))); print(f"basecall everything: >=1x {full[1][0]:.1f}%  >=2x {full[1][1]:.1f}%  >=5x {full[1][2]:.1f}% of the chromosome")
Qs = (0.0, 0.3, 0.5, 0.7)
for T_i, T in enumerate((1, 2, 5)):
    for lv in (90, 95):
        out = []
        for Q in Qs:
            ok = [r for r in res if r[0] == Q and r[4][T_i] >= lv]
            out.append(f"Q={Q}: " + (lambda b: f"{b[3]:5.1f}% bases, err {b[5]:5.2f}%")(min(ok, key=lambda r: r[3])) if ok else f"Q={Q}: not reached")
        print(f"  triage >= {T}x on {lv}%:  " + "  |  ".join(out))
def prefix_best(order_idx, T_i, lv):
    cum = np.cumsum(bases[order_idx])
    for frac in np.linspace(0.02, 1.0, 100):
        kk = int(np.searchsorted(cum, frac * tot)) + 1
        if kk > len(order_idx): break
        b, cv, er = score(list(order_idx[:kk]))
        if cv[T_i] >= lv: return b, er
    return None
print("\n  BASELINES (no overlap logic), same goals:")
for T_i, T in enumerate((1, 2, 5)):
    for lv in (90, 95):
        out = []
        for Q in Qs:
            el = np.where(qpct >= Q)[0]; r_ = prefix_best(rng.permutation(el), T_i, lv)
            out.append(f"gate Q={Q}: " + (f"{r_[0]:5.1f}% bases, err {r_[1]:5.2f}%" if r_ else "not reached"))
        fc = prefix_best(np.arange(N), T_i, lv); lg = prefix_best(np.argsort(-bases), T_i, lv)
        print(f"  >= {T}x on {lv}%:  " + "  |  ".join(out) + "  |  first-come: " + (f"{fc[0]:5.1f}%, {fc[1]:5.2f}%" if fc else "n/r") + "  |  longest-first: " + (f"{lg[0]:5.1f}%, {lg[1]:5.2f}%" if lg else "n/r"))
