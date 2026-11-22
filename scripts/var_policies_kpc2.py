"""Downstream test on KPC2 (first N reads of flow cell FAX76295, hac basecalls): does a triaged subset give the same variant-calling accuracy as basecalling everything,
and better than random / first-come subsets of the SAME size?  Selection uses only pre-basecall information (Rawsamble overlaps, noise-gate predictor frozen from E. coli R10);
estimated bases (0.08 b/sample) drive selection, REAL basecalled lengths are used to match budgets.   usage: var_policies_kpc2.py <N> <out_prefix>"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(__file__)); import var_eval_lib as V
Nn = int(sys.argv[1]); outp = sys.argv[2]; K = os.environ.get("KROOT", "/home/sgomathi/rawsamble_work/kpc2_run").rstrip("/") + "/"
IDS = os.environ.get("IDS", K + "stream/ids_first8000.txt"); FQ = os.environ.get("FQ", K + "stream/first8000_hac.fastq")
ids = [l.strip() for l in open(IDS)][:Nn]; S = set(ids)
fq = {}
with open(FQ) as f:
    rec = []
    for l in f:
        rec.append(l)
        if len(rec) == 4:
            fq[rec[0][1:].split()[0]] = "".join(rec); rec = []
ids = [r for r in ids if r in fq]; S = set(ids); print(f"{len(ids)} of the first {Nn} reads are basecalled", flush=True)
# filtered PAF so that triage only sees these reads
fp = K + "ds/online_sub.paf"
kept = {}; qlen0 = {}
for l in open(K + "run/online.paf"):
    c = l.split("\t")
    if c[0] not in S: continue
    kept.setdefault(c[0], []); qlen0.setdefault(c[0], c[1])
    if c[5] != "*" and c[5] in S: kept[c[0]].append(l)
with open(fp, "w") as o:                           # stream order = first appearance in the original PAF; reads without a retained overlap get a stub record
    for q, ls in kept.items():
        if ls: o.writelines(ls)
        else: o.write(f"{q}\t{qlen0[q]}\t*\t*\t*\t*\t*\t*\t*\t*\t*\t0\tmt:f:0\tci:i:1\tsl:i:0\tcm:i:0\tnc:i:0\ts1:i:0\tsm:f:0\n")
os.environ["CONTIG"] = "x"; os.environ["TRI_W"] = "1000"
sys.argv = [sys.argv[0], fp, fp, "3", "0.25"]
src = open(os.path.dirname(__file__) + "/triage_sim.py").read().split('print(f"stream:')[0]
ns = {"__file__": __file__}; exec(compile(src, "ts", "exec"), ns)
names, N, recs, order, qlen = (ns[k] for k in ("names", "N", "recs", "order", "qlen")); W = ns["W"]
real = np.array([len(fq[r].split("\n")[1]) for r in names]); est = pd.read_csv(K + "stream/order.tsv", sep="\t").set_index("read_id").n_samples.reindex(names).values * 0.08
R = "/home/sgomathi/rawsamble_work/r10_run/"; F = ["n_samples", "sd", "mad_diff", "med_pa", "frac_hi"]
tr = pd.read_csv(R + "sigfeat_60k.tsv", sep="\t"); idn = {}
for l in open(R + "truth_c_60k.paf"):
    c = l.split("\t")
    if c[0] not in idn and int(c[11]) >= 20: idn[c[0]] = int(c[9]) / int(c[10])
tr = tr[tr.read.isin(idn)]; model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[F], tr.read.map(idn))
sf = pd.read_csv(K + "stream/sigfeat.tsv", sep="\t").set_index("read").reindex(names); qpct = (rankdata(model.predict(sf[F])) - 1) / (N - 1)
def triage(D, tau, Q):
    chosen = np.zeros(N, bool)
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
        if np.maximum(0, D - np.cumsum(diff)[:nW]).mean() / D >= tau: chosen[i] = True
    return np.where(chosen)[0]
def evaluate(sel, tag):
    d = K + f"ds/{outp}_{tag}"; os.makedirs(d, exist_ok=True); p = d + "/reads.fastq"
    with open(p, "w") as o: o.writelines(fq[names[i]] for i in sel)
    r = V.score(p, d, threads=8); r["bases_Mb"] = round(real[sel].sum() / 1e6, 1); return r
def at_bases(order_idx, target): return order_idx[: int(np.searchsorted(np.cumsum(real[order_idx]), target)) + 1]
rng = np.random.default_rng(1); rows = []
def add(label, r): rows.append((label, r)); print(f"{label:<38} {r['bases_Mb']:6.1f} Mb  depth {r['mean_depth']:5.1f}x | SNP P {r['SNP']['P']:6.2f} R {r['SNP']['R']:6.2f} F1 {r['SNP']['F1']:6.2f} | indel F1 {r['indel']['F1']:6.2f}", flush=True)
allsel = np.arange(N); add("ALL reads (basecall everything)", evaluate(allsel, "all"))
for D, Q in ((3, 0.0), (3, 0.3), (5, 0.0), (5, 0.3)):
    sel = triage(D, 0.25, Q); b = real[sel].sum(); tag = f"triage_D{D}_Q{Q}"; add(f"TRIAGE D={D} Q={Q}", evaluate(sel, tag))
    fc = at_bases(np.arange(N), b); add(f"  first-come, same bases", evaluate(fc, tag + "_fc"))
    rr = [evaluate(at_bases(rng.permutation(N), b), tag + f"_rnd{k}") for k in range(3)]
    add("  random, same bases (mean of 3)", {"bases_Mb": np.mean([x["bases_Mb"] for x in rr]).round(1), "mean_depth": np.mean([x["mean_depth"] for x in rr]).round(1), "SNP": {m: np.mean([x["SNP"][m] for x in rr]).round(2) for m in ("P", "R", "F1")}, "indel": {"F1": np.mean([x["indel"]["F1"] for x in rr]).round(2)}})
    if Q > 0:
        el = np.where(qpct >= Q)[0]; rg = [evaluate(at_bases(rng.permutation(el), b), tag + f"_gate{k}") for k in range(3)]
        add("  gate-only random, same bases (mean of 3)", {"bases_Mb": np.mean([x["bases_Mb"] for x in rg]).round(1), "mean_depth": np.mean([x["mean_depth"] for x in rg]).round(1), "SNP": {m: np.mean([x["SNP"][m] for x in rg]).round(2) for m in ("P", "R", "F1")}, "indel": {"F1": np.mean([x["indel"]["F1"] for x in rg]).round(2)}})
print("\n-- depth ladder: random subsets of all reads at fixed fractions (3 draws each) --")
for fr in (0.2, 0.35, 0.5, 0.75):
    rr = [evaluate(at_bases(rng.permutation(N), fr * real.sum()), f"ladder{fr}_{k}") for k in range(3)]
    add(f"random {int(fr*100)}% of bases", {"bases_Mb": np.mean([x["bases_Mb"] for x in rr]).round(1), "mean_depth": np.mean([x["mean_depth"] for x in rr]).round(1), "SNP": {m: np.mean([x["SNP"][m] for x in rr]).round(2) for m in ("P", "R", "F1")}, "indel": {"F1": np.mean([x["indel"]["F1"] for x in rr]).round(2)}})
