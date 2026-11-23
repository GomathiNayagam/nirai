"""Would mapping only every k-th 1000-event window of a read give the same triage decisions? Replay on an existing overlap PAF: the deficit is computed from every k-th window only.
Reports agreement with the full-window decisions and bases chosen.  usage: triage_window_stride_test.py <online.paf> [D=10]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.triage import read_paf, window_depth, length_recall, EVENTS_PER_BASE
paf = sys.argv[1]; D = int(sys.argv[2]) if len(sys.argv) > 2 else 10
names, qlen, recs = read_paf(paf); idx = {r: i for i, r in enumerate(names)}; L = np.array([qlen[n] for n in names]) / EVENTS_PER_BASE
def run(k):
    ch = np.zeros(len(names), bool)
    for i, r in enumerate(names):
        nb = L[i]
        if nb < 1000: ch[i] = True; continue
        e = window_depth(qlen[r], recs[r], ch, idx, 1000)[::k]
        e = np.minimum(e / length_recall(nb), D) if D >= 5 else e
        if np.maximum(0, D - e).mean() / D >= 0.25: ch[i] = True
    return ch
full = run(1); print(f'{paf.split("/")[-2]}: {len(names)} reads, D={D}; full windows: {L[full].sum()/L.sum()*100:.1f}% of bases chosen')
for k in (2, 3, 5, 10):
    c = run(k); print(f'  every {k:>2}th window (~{100//k}% of mapping): {L[c].sum()/L.sum()*100:5.1f}% of bases chosen, decision agreement {100*(c==full).mean():5.1f}%, chosen-but-not-full {int((c&~full).sum())}, full-but-not-chosen {int((~c&full).sum())}', flush=True)
