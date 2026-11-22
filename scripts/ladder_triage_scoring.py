"""Reference-free scoring of Nirai triage on the DNA ladder.
Decisions (Nirai): frozen triage rule (D=3,5,10; tau 0.25; window 1000 events) from signal overlaps + signal length only (same inputs as the leakage audit).
Scoring truth built AFTER the decisions and never given to the rule: Dorado hac basecalls of all 60k reads.
  fragment 'types' = connected components of near-full-length minimap2 overlaps among the 1,500-read basecalled sample (no reference); one representative read per type;
  every basecalled read is assigned to a type by mapping to the representatives (type depth = reads covering >=80% of the representative).
Question: does the triaged subset keep every fragment type at enough depth, with fewer bases than first-come / random subsets of the same size?
usage: ladder_triage_scoring.py"""
import os, sys, subprocess, numpy as np, pandas as pd
from collections import defaultdict, Counter
L = os.path.expanduser('~/rawsamble_work/ladder/'); S = L + 'stream/'; OUT = L + 'triage_scoring/'; os.makedirs(OUT, exist_ok=True)
order = pd.read_csv(S + 'order.tsv', sep='\t'); nsamp = dict(zip(order.read_id, order.n_samples))
seq = {}
for part in ('part_00', 'part_01', 'part_02'):
    with open(S + part + '.fastq') as f:
        while True:
            h = f.readline()
            if not h: break
            seq[h[1:].split()[0]] = f.readline().strip(); f.readline(); f.readline()
print(f'basecalled reads: {len(seq)} of {len(order)} in the stream', flush=True)
# ---- types from the 1,500 sample (reference-free) ----
sm = {}
with open(S + 's1500_hac.fastq') as f:
    while True:
        h = f.readline()
        if not h: break
        sm[h[1:].split()[0]] = len(f.readline().strip()); f.readline(); f.readline()
par = {r: r for r in sm}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for l in open(S + 's1500_ava.paf'):
    c = l.split('\t')
    if c[0] == c[5] or c[0] not in sm or c[5] not in sm: continue
    span = min(int(c[3]) - int(c[2]), int(c[8]) - int(c[7])); sh = min(int(c[1]), int(c[6]))
    if span >= 0.8 * sh and int(c[9]) >= 0.8 * span * 0.8: par[find(c[0])] = find(c[5])
comp = defaultdict(list)
for r in sm: comp[find(r)].append(r)
types = [rs for rs in comp.values() if len(rs) >= 3]
reps = {}
with open(OUT + 'reps.fa', 'w') as o:
    for k, rs in enumerate(types):
        rs = sorted(rs, key=lambda r: sm[r]); rep = rs[len(rs) // 2]; reps[f'T{k}'] = sm[rep]
        # sequence of the representative from the part fastq (sample reads are a subset of the stream)
        o.write(f'>T{k}\n{seq[rep]}\n' if rep in seq else '')
print(f'types (components with >=3 reads in the 1,500 sample): {len(types)}; reads in them {sum(len(r) for r in types)} of {len(sm)}', flush=True)
reps = {k: v for k, v in reps.items()}
allfa = OUT + 'all.fa'
with open(allfa, 'w') as o:
    for r, s in seq.items(): o.write(f'>{r}\n{s}\n')
if not os.path.exists(OUT + 'map.paf'):
    subprocess.run(f'minimap2 -x map-ont -t 8 --secondary=no -N 0 -o {OUT}map.paf {OUT}reps.fa {allfa}', shell=True, check=True, stderr=subprocess.DEVNULL)
tp = {}; rl = {}
for l in open(OUT + 'map.paf'):
    c = l.split('\t'); r = c[0]
    if r in tp: continue
    qlen = int(c[1]); rl[c[5]] = int(c[6])
    if (int(c[3]) - int(c[2])) >= 0.8 * qlen: tp[r] = (c[5], (int(c[8]) - int(c[7])) / int(c[6]))
print(f'reads assigned to a type (read >=80% aligned): {len(tp)} of {len(seq)} ({100*len(tp)/len(seq):.0f}%)', flush=True)
full = {r: t for r, (t, cov) in tp.items() if cov >= 0.8}
tcount = Counter(full.values()); print(f'reads covering >=80% of their type representative: {len(full)} ({100*len(full)/len(seq):.0f}%)')
print('type sizes (full-length reads, whole stream): ' + ', '.join(f'{t}:{n}' for t, n in sorted(tcount.items(), key=lambda x: -x[1])[:20]) + (' ...' if len(tcount) > 20 else ''))
# ---- Nirai decisions (signal-only inputs) ----
paf = L + 'run/online.paf'; names_o = list(order.read_id)
orderi, qlen_o, recs = {}, {}, defaultdict(list)
for l in open(paf):
    c = l.split('\t', 6); q = c[0]
    if q not in orderi: orderi[q] = len(orderi); qlen_o[q] = int(c[1])
    if c[5] != '*': recs[q].append((c[5], int(c[2]), int(c[3])))
names = sorted(orderi, key=orderi.get); N = len(names); idx = {r: i for i, r in enumerate(names)}
real = np.array([len(seq.get(r, '')) for r in names]); have = real > 0
def decide(D, W=1000, tau=0.25):
    chosen = np.zeros(N, bool)
    for i in range(N):
        r = names[i]; nW = max(1, -(-qlen_o[r] // W)); diff = np.zeros(nW + 1, np.int32); by_t = defaultdict(list)
        for t, qs, qe in recs[r]:
            j = idx.get(t)
            if j is not None and j < i and chosen[j]: by_t[t].append((qs, qe))
        for t, iv in by_t.items():
            iv.sort(); cs, ce = iv[0]
            for s, e in iv[1:] + [(None, None)]:
                if s is not None and s <= ce: ce = max(ce, e); continue
                lo, hi = cs // W, min(nW, -(-ce // W))
                if hi > lo: diff[lo] += 1; diff[hi] -= 1
                if s is not None: cs, ce = s, e
        if np.maximum(0, D - np.cumsum(diff)[:nW]).mean() / D >= tau: chosen[i] = True
    return chosen
rng = np.random.default_rng(1)
def type_depths(sel_idx):
    c = Counter(); 
    for i in sel_idx:
        t = full.get(names[i])
        if t: c[t] += 1
    return c
def line(label, sel, D):
    c = type_depths(sel); b = real[sel].sum(); ok = [t for t in tcount if tcount[t] >= D]
    kept = lambda k: np.mean([c.get(t, 0) >= k for t in ok]) * 100 if ok else float('nan')
    return f'{label:<28} {100*b/real.sum():5.1f}% bases {100*len(sel)/N:5.1f}% reads | types with >=1 / >=3 / >=D({D}) full-length reads: {kept(1):5.1f}% / {kept(3):5.1f}% / {kept(D):5.1f}%  (of {len(ok)} types with >=D in the full stream)'
print(f'\nALL reads: {line("all", np.arange(N), 3).split("|")[0]}')
for D in (3, 5, 10):
    ch = decide(D); sel = np.where(ch)[0]; b = real[sel].sum()
    print(f'\n== D={D} ==\n' + line('NIRAI triage', sel, D), flush=True)
    cum = np.cumsum(real); fc = np.arange(int(np.searchsorted(cum, b)) + 1); print(line('first-come, same bases', fc, D))
    rr = []
    for k in range(5):
        perm = rng.permutation(N); cp = np.cumsum(real[perm]); rr.append(perm[: int(np.searchsorted(cp, b)) + 1])
    ok = [t for t in tcount if tcount[t] >= D]; m = lambda k: np.mean([np.mean([type_depths(s).get(t, 0) >= k for t in ok]) for s in rr]) * 100
    print(f'{"random, same bases (mean of 5)":<28} {100*b/real.sum():5.1f}% bases | types with >=1 / >=3 / >=D({D}) full-length reads: {m(1):5.1f}% / {m(3):5.1f}% / {m(D):5.1f}%')
