"""What does the triage step see?  Probe of the overlap-based redundancy rule on melb2 (first 50,000 reads, public hac calls give the true reference position of each read).
Questions: (1) how good is the depth estimate by read length / quality / repeat, (2) which true overlaps does Rawsamble find (recall by overlap length), which ones it reports are false,
(3) where does the rule disagree with an oracle that knows the true depth, (4) which read properties explain the decision (single-feature AUC).
Output: results/nirai_triage_probe_melb2.txt.   usage: triage_probe.py [D]"""
import os, sys, subprocess, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from nirai.triage import read_paf, window_depth, length_recall, EVENTS_PER_BASE
W = os.path.expanduser('~/rawsamble_work/melb2/'); REF = W + 'truth/ATCC_17802__202309/'
D = int(sys.argv[1]) if len(sys.argv) > 1 else 5; TAU = 0.25; WIN = 1000; BIN = 100
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_triage_probe_melb2.txt'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ---- true position of each read (minimap2 on the public hac calls)
fq = W + 'fc50k/stream/first50000_hac.fastq'; paf_t = W + 'fc50k/run/probe_truth.paf'
if not os.path.exists(paf_t): subprocess.run(f'minimap2 -x map-ont -t 8 --secondary=no {REF}reference.fna {fq} > {paf_t} 2> {paf_t}.err', shell=True, check=True)
best = {}
for l in open(paf_t):
    c = l.split('\t'); r = c[0]; m = int(c[9])
    if r not in best or m > best[r][0]: best[r] = (m, c[4], c[5], int(c[7]), int(c[8]), int(c[11]), int(c[1]), int(c[3]) - int(c[2]), int(c[10]))
Q = {}
with open(fq) as f:
    while True:
        h = f.readline()
        if not h: break
        f.readline(); f.readline(); q = np.frombuffer(f.readline().strip().encode(), np.uint8).astype(float) - 33; Q[h[1:].split()[0]] = -10 * np.log10(np.mean(10 ** (-q / 10)))
rep = collections.defaultdict(list)
for l in open(REF + 'ATCC_17802__202309.repetitive_regions.bed'): c = l.split(); rep[c[0]].append((int(c[1]), int(c[2])))
def rep_frac(ch, a, b):
    return sum(max(0, min(b, e) - max(a, s)) for s, e in rep.get(ch, [])) / max(1, b - a)

# ---- replay the rule, keep what it saw
names, qlen, recs = read_paf(W + 'fc50k/run/online.paf'); N = len(names); idx = {r: i for i, r in enumerate(names)}
chroms = sorted({b[2] for b in best.values()}); clen = {}
for l in open(REF + 'reference.fna.fai'): c = l.split(); clen[c[0]] = int(c[1])
cov = {(c, s): np.zeros(clen[c] // BIN + 2, np.int32) for c in clen for s in '+-'}
chosen = np.zeros(N, bool); nb = np.array([qlen[r] / EVENTS_PER_BASE for r in names])
e_raw = np.zeros(N); e_cor = np.zeros(N); t_dep = np.full(N, np.nan); orc = np.zeros(N, bool); has_t = np.zeros(N, bool); novl = np.zeros(N); ovl_frac = np.zeros(N)
for i, r in enumerate(names):
    b = best.get(r); e = window_depth(qlen[r], recs[r], chosen, idx, WIN); e_raw[i] = e.mean(); novl[i] = len(recs[r])
    nbi = nb[i]
    if nbi < 1000: chosen[i] = True; e_cor[i] = np.nan
    else:
        ec = np.minimum(e / length_recall(nbi), D); e_cor[i] = ec.mean(); chosen[i] = np.maximum(0, D - ec).mean() / D >= TAU
    ovl_frac[i] = (e > 0).mean()
    if b is not None and b[7] >= 0.5 * b[6]:      # mapped, at least half of read aligned
        has_t[i] = True; _, st, ch, ts, te, mq, ql, al, _ = b; nW = max(1, -(-qlen[r] // WIN)); k = (np.arange(nW) + .5) * WIN / qlen[r]; k = np.clip(k, 0, 1)
        pos = ts + k * (te - ts) if st == '+' else te - k * (te - ts); c = cov[(ch, st)]; td = c[np.clip((pos // BIN).astype(int), 0, len(c) - 1)]
        t_dep[i] = td.mean(); orc[i] = np.maximum(0, D - np.minimum(td, D)).mean() / D >= TAU if nbi >= 1000 else True
    if chosen[i] and b is not None:
        _, st, ch, ts, te, *_ = b; cov[(ch, st)][ts // BIN: te // BIN + 1] += 1
mapped = np.array([r in best for r in names]); qv = np.array([Q.get(r, np.nan) for r in names]); L = nb
rf = np.array([rep_frac(best[r][2], best[r][3], best[r][4]) if r in best else np.nan for r in names]); mq = np.array([best[r][5] if r in best else np.nan for r in names])
P(f'melb2 first {N} reads, D={D}, tau={TAU}, window={WIN} events, length-corrected; truth = minimap2 of public hac calls')
P(f'reads {N}; with public call {int((~np.isnan(qv)).sum())}; used as truth (aligned >=50%) {int(has_t.sum())}; chosen for basecalling {int(chosen.sum())} ({100*chosen.mean():.1f}%), bases chosen {100*nb[chosen].sum()/nb.sum():.1f}%')
bins = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 1e9)]; lab = lambda a, b: f'{a/1000:g}-{"inf" if b > 1e8 else f"{b/1000:g}"}kb'

P('\n== 1. depth estimate vs true depth (same-strand coverage by the reads the rule chose earlier; per-read mean over 1-kb windows) ==')
P('length      n   chosen%  true_depth  est_raw  est/true  est_corrected | oracle_choose%  agree%  wasted%(chosen,oracle skips)  lost%(skipped,oracle chooses)')
for a, b in bins:
    m = (L >= a) & (L < b); h = m & has_t
    if h.sum() < 20: continue
    ag = (chosen[h] == orc[h]).mean() * 100; w = (chosen[h] & ~orc[h]).mean() * 100; lo = (~chosen[h] & orc[h]).mean() * 100
    P(f'{lab(a,b):<9} {int(m.sum()):>6} {100*chosen[m].mean():7.1f}  {np.nanmean(t_dep[h]):9.2f} {e_raw[h].mean():8.2f} {e_raw[h].mean()/max(1e-9,np.nanmean(t_dep[h])):8.2f} {np.nanmean(e_cor[h]) if a>=1000 else float("nan"):12.2f} | {100*orc[h].mean():10.1f} {ag:9.1f} {w:14.1f} {lo:22.1f}')
h = has_t & (L >= 1000); P(f'Spearman est_raw vs true depth (reads >=1 kb): {spearmanr(e_raw[h], t_dep[h])[0]:.3f}; overall agreement with oracle {100*(chosen[h]==orc[h]).mean():.1f}%')

P('\n== 2. which true overlaps does Rawsamble find? (pairs of mapped same-strand reads, earlier read j < later read i, true overlap on the reference) ==')
ov = collections.defaultdict(list)
for i, r in enumerate(names):
    if has_t[i]:
        _, st, ch, ts, te, *_ = best[r]
        for bb in range(ts // 10000, te // 10000 + 1): ov[(ch, st, bb)].append(i)
found = {(i, idx[t]) for i, r in enumerate(names) for t, _, _ in recs[r] if t in idx}
pairs = []
for i, r in enumerate(names):
    if not has_t[i]: continue
    _, st, ch, ts, te, *_ = best[r]; seen = set()
    for bb in range(ts // 10000, te // 10000 + 1):
        for j in ov[(ch, st, bb)]:
            if j < i and j not in seen:
                seen.add(j); _, _, _, ts2, te2, *_ = best[names[j]]; o = min(te, te2) - max(ts, ts2)
                if o > 0: pairs.append((i, j, o, (i, j) in found))
pa = np.array(pairs, dtype=float); P(f'true same-strand overlapping pairs {len(pa)}; found by Rawsamble (target listed for the later read) {100*pa[:,3].mean():.1f}%')
P('recall by true overlap length:'); [P(f'  {lab(a,b):<9} pairs {int(((pa[:,2]>=a)&(pa[:,2]<b)).sum()):>8}  found {100*pa[(pa[:,2]>=a)&(pa[:,2]<b),3].mean():5.1f}%') for a, b in [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 1e9)] if ((pa[:,2]>=a)&(pa[:,2]<b)).sum() > 20]
P('recall by the shorter read of the pair (kb):')
sh = np.minimum(L[pa[:, 0].astype(int)], L[pa[:, 1].astype(int)]); [P(f'  {lab(a,b):<9} pairs {int(((sh>=a)&(sh<b)).sum()):>8}  found {100*pa[(sh>=a)&(sh<b),3].mean():5.1f}%') for a, b in bins if ((sh>=a)&(sh<b)).sum() > 20]
P('precision of reported overlaps (both reads mapped): is the reported pair a true reference overlap of >=200 bp?')
tp = {(int(a), int(b)): o for a, b, o, _ in pairs}; rows = []
for i, r in enumerate(names):
    for t, qs, qe in recs[r]:
        j = idx.get(t)
        if j is not None and has_t[i] and has_t[j]: rows.append((i, j, (qe - qs) / EVENTS_PER_BASE, tp.get((i, j), 0) >= 200, max(rep_frac(*best[names[i]][2:5]) , 0)))
rw = np.array(rows, dtype=float); P(f'  reported pairs with both reads mapped {len(rw)}; true overlap >=200 bp: {100*rw[:,3].mean():.1f}%')
for a, b in [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 1e9)]:
    m = (rw[:, 2] >= a) & (rw[:, 2] < b)
    if m.sum() > 20: P(f'  reported overlap length {lab(a,b):<9} n {int(m.sum()):>8}  true {100*rw[m,3].mean():5.1f}%')
mm = rw[:, 4] > 0.5; P(f'  query read mostly in a repeat region (>50%): n {int(mm.sum())}, true {100*rw[mm,3].mean():.1f}%; others {100*rw[~mm,3].mean():.1f}%')

P('\n== 3. what explains the decision? single-feature AUC for CHOSEN (reads >=1 kb with truth) ==')
h = has_t & (L >= 1000); feats = {'read length': L, 'mean Q': qv, 'mapq': mq, 'repeat fraction': rf, 'arrival order': np.arange(N), 'overlap records': novl, 'fraction of windows with any overlap': ovl_frac, 'estimated depth': e_raw, 'TRUE depth (hidden from rule)': t_dep}
for k, v in feats.items():
    m = h & ~np.isnan(v); P(f'  {k:<38} AUC {roc_auc_score(chosen[m], v[m]):.3f}   median chosen {np.median(v[m & chosen]):.2f} vs skipped {np.median(v[m & ~chosen]):.2f}')
P('chosen share by arrival order (how the rule saturates): ' + ', '.join(f'{a//1000}-{b//1000}k: {100*chosen[a:b].mean():.0f}%' for a, b in [(0, 5000), (5000, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000)]))

P('\n== 4. where does the rule go wrong? (reads >=1 kb with truth) ==')
w = h & chosen & ~orc; lo = h & ~chosen & orc
for nm, m in (('WASTED: chosen but true depth already enough', w), ('LOST: skipped but true depth still short', lo), ('all reads >=1 kb with truth', h)):
    P(f'  {nm:<46} n {int(m.sum()):>6}  median length {np.median(L[m]):6.0f} b  median Q {np.nanmedian(qv[m]):5.1f}  repeat frac {np.nanmean(rf[m]):.3f}  true depth {np.nanmean(t_dep[m]):5.2f}  est depth {e_raw[m].mean():5.2f}')
P('  LOST reads by true depth (they were skipped, so the estimate said "enough"): true depth <1: %d, 1-2: %d, 2-3: %d, >=3: %d' % tuple(int((lo & (t_dep >= a) & (t_dep < b)).sum()) for a, b in [(0, 1), (1, 2), (2, 3), (3, 99)]))
fin = {k: v for k, v in cov.items()}; allc = {k: np.zeros_like(v) for k, v in cov.items()}
for i, r in enumerate(names):
    if r in best: _, st, ch, ts, te, *_ = best[r]; allc[(ch, st)][ts // BIN: te // BIN + 1] += 1
tot = sum(v.sum() for v in allc.values()); ch_cov = sum(v.sum() for v in fin.values()); f1 = lambda dd, th: np.mean(np.concatenate([(dd[(c, s)] + dd[(c, '-' if s == '+' else '+')] >= th)[:clen[c] // BIN] for c in clen for s in '+']))
P(f'final coverage (both strands, 100-bp bins): share of genome at depth >=5 / >=10: all mapped reads {100*f1(allc,5):.1f}% / {100*f1(allc,10):.1f}%; chosen reads {100*f1(fin,5):.1f}% / {100*f1(fin,10):.1f}%; chosen keeps {100*ch_cov/tot:.1f}% of mapped read-bins')
