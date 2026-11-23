"""Label-free per-run threshold: score the first K reads of a NEW run with the trained gate and set the eject threshold at a percentile of THOSE scores (no labels, no basecalls).
Tests whether good-read retention >= 98% holds on a held-out run. Model = 1D CNN + scalars + kit, trained on all other non-ladder runs + extra RDH275 reads (aug), 0.4 s. Evaluated on reads after the first K.
Also reports the carried-over (training-percentile) threshold for comparison.  Reuses data prep from gate_rdh275_failaug.py."""
import os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate_rdh275_failaug.py')).read().split("notr = ")[0]
g = {'__file__': os.path.abspath(__file__), '__name__': 'prep'}; exec(compile(src, 'prep', 'exec'), g)
np, y, R, I, SRC, fit, predict = (g[k] for k in ('np', 'y', 'R', 'I', 'SRC', 'fit', 'predict')); roc = g['roc_auc_score']
notr = ~np.isin(R, ['LADDER_Oct']); extra = SRC != 'base'
print(f'\n{"held-out run":<18}{"K":>5}{"pct":>5}{"good kept":>11}{"low caught":>12}   (carried-over threshold: good kept / low caught)  ROLLING (last 1000 scores, update every 100): good kept / low caught', flush=True)
for run in ['RDH275', 'KPC2_FAX39616', 'KPC2_FAX76295', 'ECOLI_ATCC25922']:
    te_all = np.where((R == run) & ~extra)[0]; te_all = te_all[np.argsort(I[te_all])]
    tri = np.where(notr & (R != run) | (notr & extra & (run != 'RDH275')))[0]   # RDH275 case: its extra reads would leak the run, so excluded when RDH275 is held out
    if run == 'RDH275': tri = np.where(notr & (R != 'RDH275'))[0]
    m = fit(tri); ptr = predict(m, tri); p = predict(m, te_all); yt = y[te_all]
    for K in (300, 1000):
        cal, ev = p[:K], p[K:]; ye = yt[K:]
        for pc in (99, 98):
            thr = np.percentile(cal, pc); thr2 = np.percentile(ptr[y[tri] == 0], pc)
            roll = np.empty(len(ev), bool); thr_r = np.empty(len(ev))   # rolling: threshold = percentile of the last 1000 scores before each block of 100 reads
            for b in range(0, len(ev), 100):
                lo = max(0, K + b - 1000); thr_r[b:b + 100] = np.percentile(p[lo:K + b], pc)
            gk = (ev[ye == 0] <= thr_r[ye == 0]).mean(); lc = (ev[ye == 1] > thr_r[ye == 1]).mean()
            print(f'{run:<18}{K:5d}{pc:5d}{100*(ev[ye==0]<=thr).mean():10.1f}%{100*(ev[ye==1]>thr).mean():11.0f}%   ({100*(ev[ye==0]<=thr2).mean():.1f}% / {100*(ev[ye==1]>thr2).mean():.0f}%)   [eval n {len(ev)}, low {int(ye.sum())}]   ROLLING: {100*gk:.1f}% / {100*lc:.0f}%', flush=True)
