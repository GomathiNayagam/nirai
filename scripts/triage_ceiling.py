"""Ceiling test for the triage step (melb2, first 50,000 reads, public hac calls, NanoVarBench truth).
What if the overlap finder were perfect?  ORACLE rules use the true reference position of every read (minimap2 on its public basecall) instead of Rawsamble overlaps, with the same depth rule
(1-kb windows, deficit >= tau=0.25, causal: only earlier CHOSEN reads count, same strand).
Arms per D: POLICY (the real rule, nirai.triage.triage), ORACLE-long (reads <1 kb always chosen, as in POLICY), ORACLE-all (the oracle also decides short reads). Each is compared with first-come and
random subsets of the SAME real bases, scored by variant calling (var_eval_lib).   usage: triage_ceiling.py [D ...]  -> results/nirai_triage_ceiling_melb2.txt"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
W = os.path.expanduser('~/rawsamble_work/melb2/'); os.environ['TRUTH'] = W + 'truth/ATCC_17802__202309/'
sys.path.insert(0, '/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/scripts'); import var_eval_lib as V
from nirai.triage import read_paf, triage, EVENTS_PER_BASE
DS = [int(x) for x in sys.argv[1:]] or [3, 5]; TAU = 0.25; WIN = 1000; BIN = 100
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_triage_ceiling_melb2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
names, qlen, recs = read_paf(W + 'fc50k/run/online.paf')
fq = {}
with open(W + 'fc50k/stream/first50000_hac.fastq') as f:
    while True:
        h = f.readline()
        if not h: break
        s, p, q = f.readline(), f.readline(), f.readline(); fq[h[1:].split()[0]] = h + s + p + q
best = {}
for l in open(W + 'fc50k/run/probe_truth.paf'):
    c = l.split('\t'); r = c[0]; m = int(c[9])
    if r not in best or m > best[r][0]: best[r] = (m, c[4], c[5], int(c[7]), int(c[8]), int(c[1]), int(c[3]) - int(c[2]))
clen = {l.split()[0]: int(l.split()[1]) for l in open(W + 'truth/ATCC_17802__202309/reference.fna.fai')}
keep = [i for i, n in enumerate(names) if n in fq]; real = np.array([len(fq[names[i]].split('\n')[1]) for i in keep]); nb = np.array([qlen[r] / EVENTS_PER_BASE for r in names])
def oracle(D, short_always):
    cov = {(c, s): np.zeros(clen[c] // BIN + 2, np.int32) for c in clen for s in '+-'}; chosen = np.zeros(len(names), bool)
    for i, r in enumerate(names):
        b = best.get(r); ok = b is not None and b[6] >= 0.5 * b[5]
        if not ok or (short_always and nb[i] < 1000): chosen[i] = True
        else:
            _, st, ch, ts, te, *_ = b; nW = max(1, -(-qlen[r] // WIN)); k = np.clip((np.arange(nW) + .5) * WIN / qlen[r], 0, 1)
            pos = ts + k * (te - ts) if st == '+' else te - k * (te - ts); c = cov[(ch, st)]; td = c[np.clip((pos // BIN).astype(int), 0, len(c) - 1)]
            chosen[i] = np.maximum(0, D - np.minimum(td, D)).mean() / D >= TAU
        if chosen[i] and best.get(r) is not None:
            _, st, ch, ts, te, *_ = best[r]; cov[(ch, st)][ts // BIN: te // BIN + 1] += 1
    return chosen
OUTD = W + 'gate_eval/ceiling/'; os.makedirs(OUTD, exist_ok=True)
def score(sel, tag):
    d = OUTD + tag; os.makedirs(d, exist_ok=True)
    with open(d + '/reads.fastq', 'w') as o: o.writelines(fq[names[keep[j]]] for j in sel)
    r = V.score(d + '/reads.fastq', d, threads=8); r['Mb'] = real[sel].sum() / 1e6; return r
def line(label, r): P(f"{label:<40}{r['Mb']:6.1f} Mb  depth {r['mean_depth']:5.1f}x | SNP P {r['SNP']['P']:6.2f} R {r['SNP']['R']:6.2f} F1 {r['SNP']['F1']:6.2f} | indel F1 {r['indel']['F1']:6.2f}")
def at_bases(order, b): return order[: int(np.searchsorted(np.cumsum(real[order]), b)) + 1]
rng = np.random.default_rng(1)
P(f'\n=== melb2 first 50,000 reads (48,025 with public call), D={DS}, tau={TAU}; oracle = true reference positions instead of Rawsamble overlaps ===')
line('ALL reads', score(np.arange(len(keep)), 'all'))
for D in DS:
    arms = {'POLICY (real Rawsamble)': triage(names, qlen, recs, D), 'ORACLE-long (shorts always chosen)': oracle(D, True), 'ORACLE-all (oracle decides shorts too)': oracle(D, False)}
    for lab, ch in arms.items():
        sel = np.array([j for j, i in enumerate(keep) if ch[i]]); b = real[sel].sum(); tag = f"{lab.split()[0]}{D}"
        P(f'-- D={D} {lab}: chosen {100*ch.mean():.1f}% of reads')
        line(f'  {lab.split()[0]} D={D}', score(sel, tag)); line('  first-come, same bases', score(at_bases(np.arange(len(keep)), b), tag + '_fc'))
        rr = [score(at_bases(rng.permutation(len(keep)), b), f'{tag}_r{k}') for k in range(2)]
        P(f"{'  random, same bases (mean of 2)':<40}{np.mean([x['Mb'] for x in rr]):6.1f} Mb  depth {np.mean([x['mean_depth'] for x in rr]):5.1f}x | SNP F1 {np.mean([x['SNP']['F1'] for x in rr]):6.2f} | indel F1 {np.mean([x['indel']['F1'] for x in rr]):6.2f}")
P('CEILING_DONE')
