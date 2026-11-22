"""Why does triage do worse than first-come on RDH275 but better on KPC2?  Cheap checks, same code for both runs.
(1) coverage and read-length mix of the stream;  (2) how much of the true same-strand depth Rawsamble sees (estimated depth / true depth, by read length),
true depth = earlier reads aligned to the reference (signal-to-reference mapping, truth_sig.paf) over the read's own reference span.
usage: triage_underperformance_check.py NAME RUNDIR [maxreads=8000]"""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict
name, root = sys.argv[1], os.path.expanduser(sys.argv[2]).rstrip('/') + '/'; MAXN = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
od = pd.read_csv(root + 'stream/order.tsv', sep='\t'); ns = dict(zip(od.read_id, od.n_samples))
order = {}; recs = defaultdict(list); qlen = {}
for l in open(root + 'run/online.paf'):
    c = l.split('\t', 6); q = c[0]
    if q not in order:
        if len(order) >= MAXN: break
        order[q] = len(order); qlen[q] = int(c[1])
    if c[5] != '*': recs[q].append((c[5], int(c[2]), int(c[3])))
names = sorted(order, key=order.get); N = len(names)
truth = {}; tl = {}
for l in open(root + 'run/truth_sig.paf'):
    c = l.split('\t')
    if c[5] == '*' or c[0] in truth or c[0] not in order: continue
    truth[c[0]] = (c[4], c[5], int(c[7]), int(c[8])); tl[c[5]] = int(c[6])
G = sum(tl.values()); bases = np.array([ns[r] * 0.08 for r in names])
print(f'== {name}: first {N} reads of the stream ==')
print(f'genome {G/1e6:.2f} Mb; stream {bases.sum()/1e6:.1f} Mb = {bases.sum()/G:.1f}x raw depth; reads with a reference position: {len(truth)} ({100*len(truth)/N:.0f}%)')
print(f'read length (est.): median {np.median(bases):.0f} b, mean {bases.mean():.0f} b; reads <1 kb: {100*(bases<1000).mean():.0f}% of reads, {100*bases[bases<1000].sum()/bases.sum():.1f}% of bases; reads <2 kb: {100*(bases<2000).mean():.0f}% / {100*bases[bases<2000].sum()/bases.sum():.1f}%')
nn = np.array([len({t for t, _, _ in recs[r]}) for r in names]); print(f'Rawsamble finds overlaps to earlier reads for {100*(nn>0).mean():.0f}% of reads; median #earlier reads overlapped {np.median(nn):.0f}')
BIN = 100; cov = {s: defaultdict(lambda: np.zeros(1, np.int32)) for s in '+-'}
covarr = {}
rows = []
for i, r in enumerate(names):
    if r in truth:
        st, ct, a, b = truth[r]; key = (st, ct)
        if key not in covarr: covarr[key] = np.zeros(tl[ct] // BIN + 2, np.int32)
        if i >= MAXN // 3:
            tr = covarr[key][a // BIN: b // BIN + 1]; true_d = float(tr.mean()) if len(tr) else 0.0
            by_t = defaultdict(list)
            for t, qs, qe in recs[r]: by_t[t].append((qs, qe))
            span = max(1, qlen[r]); est = 0.0
            for t, iv in by_t.items():
                iv.sort(); cs, ce = iv[0]; tot = 0
                for s, e in iv[1:] + [(None, None)]:
                    if s is not None and s <= ce: ce = max(ce, e); continue
                    tot += ce - cs
                    if s is not None: cs, ce = s, e
                est += tot / span
            rows.append((bases[i], true_d, est))
        covarr[key][a // BIN: b // BIN + 1] += 1
d = pd.DataFrame(rows, columns=['len', 'true', 'est']); d = d[d.true >= 3]
d['lb'] = pd.cut(d.len, [0, 1000, 2000, 4000, 8000, 1e9], labels=['<1kb', '1-2kb', '2-4kb', '4-8kb', '>8kb'])
print(f'\nreads in the later 2/3 of the stream with true same-strand depth >= 3: {len(d)}; overall estimated/true depth = {d.est.sum()/d.true.sum():.2f}')
print('by read length:      reads  est/true depth  (share of reads whose estimated depth >= half the true depth)')
for k, g in d.groupby('lb', observed=True): print(f'  {k:>6}: {len(g):8d}   {g.est.sum()/g.true.sum():8.2f}   {100*(g.est>=0.5*g.true).mean():6.0f}%')
