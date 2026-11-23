"""Barcode-aware triage test. Same isolate in several 'barcodes': the 4 KPC2 flow cells (pooled.pod5, interleaved) stand in for 4 barcoded samples of the SAME genome.
Global triage (no groups) sees all four as one sample, so later barcodes look redundant and are starved; group-aware triage brings each barcode to depth D on its own.
Reports per-barcode bases kept, mean depth and % of genome >= 10x (from reference spans of the basecalls; never used by the rule).  usage: barcode_triage_test.py [D=10]"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.triage import read_paf, triage, EVENTS_PER_BASE
P = '/home/sgomathi/rawsamble_work/kpc2_pool/'; D = int(sys.argv[1]) if len(sys.argv) > 1 else 10
o = pd.read_csv(P + 'order.tsv', sep='\t'); fc = dict(zip(o.read_id, o.flowcell))
names, qlen, recs = read_paf(P + 'online.paf'); L = np.array([qlen[n] for n in names]) / EVENTS_PER_BASE; G = np.array([fc[n] for n in names])
T = {}
for l in open(P + 'truth_bc.paf'):
    c = l.split('\t')
    if c[0] not in T and int(c[11]) >= 20 and int(c[10]) >= 0.5 * int(c[1]): T[c[0]] = (int(c[7]), int(c[8]))
GL = 5400000
def cov(sel):
    cv = np.zeros(GL // 100 + 2, np.int32)
    for i in sel:
        t = T.get(names[i])
        if t: cv[t[0] // 100: t[1] // 100 + 1] += 1
    return cv[: GL // 100]
def show(label, ch):
    out = f'{label:<30}'
    for g in sorted(set(G)):
        m = (G == g); sel = np.where(ch & m)[0]; c = cov(sel); out += f' | {g}: kept {L[sel].sum()/L[m].sum()*100:3.0f}% depth {c.mean():4.1f}x >=10x {100*(c>=10).mean():3.0f}%'
    print(out, flush=True)
print(f'{len(names)} reads, barcodes (flow cells): ' + ', '.join(f'{g} {int((G==g).sum())}' for g in sorted(set(G))))
show('ALL reads', np.ones(len(names), bool))
show(f'global triage D={D}', triage(names, qlen, recs, D))
show(f'barcode-aware triage D={D}', triage(names, qlen, recs, D, groups=G))
