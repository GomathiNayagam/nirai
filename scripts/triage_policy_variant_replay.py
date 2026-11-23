"""Variant-calling replay of the packaged triage policy (nirai.triage.triage: reads <1 kb always basecalled, length-corrected depth for D>=5) on KPC2 pooled data.
Compares: ALL reads; ORIGINAL rule (no short-read policy, no length correction); POLICY rule; and random / first-come subsets of the same real basecalled bases.
Needs kpc2_pool/{pooled.fastq, online.paf}; frozen D, tau=0.25.   usage: triage_policy_variant_replay.py [D ...]  (default 5 10)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
import var_eval_lib as V
from nirai.triage import read_paf, triage
P = '/home/sgomathi/rawsamble_work/kpc2_pool/'; DS = [int(x) for x in sys.argv[1:]] or [5, 10]
fq = {}
with open(P + 'pooled.fastq') as f:
    rec = []
    for l in f:
        rec.append(l)
        if len(rec) == 4: fq[rec[0][1:].split()[0]] = ''.join(rec); rec = []
names, qlen, recs = read_paf(P + 'online.paf'); keep = [i for i, n in enumerate(names) if n in fq]
real = np.array([len(fq[names[i]].split('\n')[1]) for i in keep]); print(f'{len(keep)} reads in stream order with basecalls, {real.sum()/1e6:.0f} Mb', flush=True)
# triage sees the full stream (as live); evaluation uses only reads that have basecalls
def score(sel, tag):
    d = P + 'ds/policy_' + tag; os.makedirs(d, exist_ok=True)
    with open(d + '/reads.fastq', 'w') as o: o.writelines(fq[names[keep[i]]] for i in sel)
    r = V.score(d + '/reads.fastq', d, threads=16); r['Mb'] = real[sel].sum() / 1e6; return r
def line(label, r): print(f"{label:<44}{r['Mb']:6.0f} Mb  depth {r['mean_depth']:5.1f}x | SNP F1 {r['SNP']['F1']:6.2f} | indel P {r['indel']['P']:6.2f} R {r['indel']['R']:6.2f} F1 {r['indel']['F1']:6.2f}", flush=True)
def at_bases(order, b): return order[: int(np.searchsorted(np.cumsum(real[order]), b)) + 1]
rng = np.random.default_rng(1); line('ALL reads (basecall everything)', score(np.arange(len(keep)), 'all'))
long = np.where(real >= 1000)[0]; line('ALL reads >= 1 kb only (no triage)', score(long, 'long_only'))
for D in DS:
    for lab, kw in (('ORIGINAL', {'min_bases': 0, 'length_correct': False}), ('POLICY', {}), ('DROPSHORT', {'short_reads': 'drop'})):
        if lab not in os.environ.get('ARMS', 'ORIGINAL,POLICY,DROPSHORT').split(','): continue
        ch = triage(names, qlen, recs, D, **kw); sel = np.array([j for j, i in enumerate(keep) if ch[i]]); b = real[sel].sum()
        line(f'{lab} D={D}', score(sel, f'{lab}{D}'))
        rr = [score(at_bases(rng.permutation(len(keep)), b), f'{lab}{D}_r{k}') for k in range(3)]
        m = {k: np.mean([x['indel'][k] for x in rr]) for k in ('P', 'R', 'F1')}
        print(f"{'   random, same bases (mean of 3)':<44}{np.mean([x['Mb'] for x in rr]):6.0f} Mb  depth {np.mean([x['mean_depth'] for x in rr]):5.1f}x | SNP F1 {np.mean([x['SNP']['F1'] for x in rr]):6.2f} | indel P {m['P']:6.2f} R {m['R']:6.2f} F1 {m['F1']:6.2f}", flush=True)
        line('   first-come, same bases', score(at_bases(np.arange(len(keep)), b), f'{lab}{D}_fc'))
