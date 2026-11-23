"""Triage on an in-silico mock metagenome (default KPC2 12,000 reads, RDH275 8,000, E. coli 1,500; MIXDIR env selects another mix; random arrival order; built by metagenome_mix_build.py).
Triage sees only the Rawsamble overlap PAF of the mixed stream (no source labels, no basecalls). Truth (never used by the rule): each read's source and reference span.
For each rule the same total (estimated) bases are given to first-come and to random subsets; reported per organism: share of the organism's bases kept, mean depth, % of genome >= 5x / 10x.
usage: metagenome_triage_test.py"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.triage import read_paf, triage, EVENTS_PER_BASE
B = '/home/sgomathi/rawsamble_work/'; MIX = B + os.environ.get('MIXDIR', 'meta_mix') + '/'
order = pd.read_csv(MIX + 'order.tsv', sep='\t'); src = dict(zip(order.read_id, order.source))
names, qlen, recs = read_paf(MIX + 'online.paf'); N = len(names); L = np.array([qlen[n] for n in names], float) / EVENTS_PER_BASE
S = np.array([src[n] for n in names]); print(f'{N} reads in the PAF stream; sources: ' + ', '.join(f'{k} {int((S==k).sum())}' for k in np.unique(S)), flush=True)
GEN = {'KPC2': {'chromosome': 5399985, 'plasmid_1': 207198, 'plasmid_2': 43380, 'plasmid_3': 13841}, 'RDH275': {'chromosome': 1793507},
       'ECOLI': {'CP149500.1': 5132220, 'CP149501.1': 48488, 'CP149502.1': 24185, 'CP149503.1': 3173, 'CP149504.1': 1919}}
def spans(paf, who, basecall):
    d = {}
    for l in open(paf):
        c = l.split('\t')
        if c[5] == '*' or c[0] in d or c[0] not in src or src[c[0]] != who: continue
        if int(c[11]) >= 20 and (not basecall or int(c[10]) >= 0.5 * int(c[1])) and c[5] in GEN[who]: d[c[0]] = (c[5], int(c[7]), int(c[8]))
    return d
T = {'KPC2': spans(B + 'kpc2_pool/truth_bc.paf', 'KPC2', True), 'RDH275': spans(B + 'rdh275/run/truth_sig_clean.paf', 'RDH275', False), 'ECOLI': spans(B + 'r10_run/truth_c_60k.paf', 'ECOLI', True)}
print('reads with a reference span:', {k: len(v) for k, v in T.items()}, flush=True)
def cover(sel, who):
    cv = {c: np.zeros(g // 100 + 1, np.int32) for c, g in GEN[who].items()}
    for i in sel:
        t = T[who].get(names[i])
        if t: c, a, b = t; cv[c][a // 100: b // 100 + 1] += 1
    return np.concatenate(list(cv.values()))
def report(label, sel, tot):
    sel = np.asarray(sel); out = f'{label:<34}{L[sel].sum()/tot*100:6.1f}% of bases |'
    for who in ('KPC2', 'RDH275', 'ECOLI'):
        m = S[sel] == who; c = cover(sel[m], who); out += f' {who} kept {L[sel][m].sum()/L[S==who].sum()*100:4.0f}% depth {c.mean():5.1f}x >=5x {100*(c>=5).mean():3.0f}% >=10x {100*(c>=10).mean():3.0f}% |'
    print(out, flush=True)
tot = L.sum(); allsel = np.arange(N); print()
report('ALL reads', allsel, tot)
rng = np.random.default_rng(1)
for D in (5, 10):
    for lab, kw in (('original', {'min_bases': 0, 'length_correct': False}), ('policy', {}), ('policy + drop <1kb', {'short_reads': 'drop'})):
        ch = triage(names, qlen, recs, D, **kw); sel = np.where(ch)[0]; b = L[sel].sum()
        print(f'\n-- D={D} {lab} --'); report(f'triage D={D} {lab}', sel, tot)
        report('  first-come, same bases', np.arange(int(np.searchsorted(np.cumsum(L), b)) + 1), tot)
        rr = [rng.permutation(N) for _ in range(5)]
        for k in (0,): report('  random, same bases (draw 1 of 5)', rr[k][: int(np.searchsorted(np.cumsum(L[rr[k]]), b)) + 1], tot)
