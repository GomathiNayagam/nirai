"""Does a longer look at the signal (0.4 s -> 0.5, 0.6, 0.8, 1.0 s) catch more low-quality reads across runs?
Same reads for every prefix (reads with >= 1.0 s of signal), same leave-one-run-out test; the October ladder is never used for training.
Labels: low = mean read Q (our hac basecalls) < QCUT.   usage: gate_prefix_sweep.py OUTDIR"""
import os, sys, numpy as np, pandas as pd, pod5
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.features import read_pa, signal_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
W = os.path.expanduser('~/rawsamble_work/'); out = sys.argv[1]; os.makedirs(out, exist_ok=True)
RUNS = {
 'RDH275': ([W+'rdh275/stream/first8000.pod5'], [W+'rdh275/stream/first8000_hac.fastq']),
 'KPC2_FAX76295': ([W+'kpc2_run/stream/first8000.pod5'], [W+'kpc2_run/stream/first8000_hac.fastq']),
 'KPC2_FAV69578': ([W+'kpc2_fc/FAV69578/stream/first8000.pod5'], [W+'kpc2_fc/FAV69578/stream/first8000_hac.fastq']),
 'KPC2_FAX39616': ([W+'kpc2_fc/FAX39616/stream/first8000.pod5'], [W+'kpc2_fc/FAX39616/stream/first8000_hac.fastq']),
 'KPC2_FAX39617': ([W+'kpc2_fc/FAX39617/stream/first8000.pod5'], [W+'kpc2_fc/FAX39617/stream/first8000_hac.fastq']),
 'LADDER_Oct': ([W+'ladder/stream/part_00.pod5', W+'ladder/stream/part_01.pod5'], [W+'ladder/stream/part_00.fastq', W+'ladder/stream/part_01.fastq']),
 'ZYMO': ([W+'zymo/stream/s3000.pod5'], [W+'zymo/stream/s3000_hac.fastq']),
}
NOTRAIN = ['LADDER_Oct']
PRE = {'0.4s': 2000, '0.5s': 2500, '0.6s': 3000, '0.8s': 4000, '1.0s': 5000}; MAXN = max(PRE.values())
def readq(fqs):
    q = {}
    for fq in fqs:
        with open(fq) as f:
            while True:
                h = f.readline()
                if not h: break
                f.readline(); f.readline(); a = np.frombuffer(f.readline().strip().encode(), np.uint8).astype(float) - 33
                if len(a): q[h[1:].split()[0]] = -10 * np.log10(np.mean(10 ** (-a / 10)))
    return q
cache = out + '/prefix_features.tsv'
if os.path.exists(cache): T = pd.read_csv(cache, sep='\t')
else:
    rows = []; ntot = {}
    for run, (ps, fqs) in RUNS.items():
        q = readq(fqs); k = 0; ntot[run] = [0, 0]
        for p in ps:
            with pod5.Reader(p) as r:
                for rec in r.reads():
                    rid = str(rec.read_id)
                    if rid not in q: continue
                    ntot[run][0] += 1; x = read_pa(rec)
                    if len(x) < MAXN: continue
                    ntot[run][1] += 1; row = [run, k, rid, q[rid]]
                    for n in PRE.values(): row += signal_features(x[:n])
                    rows.append(row); k += 1
        print(run, 'reads with basecalls', ntot[run][0], 'with >=1.0 s of signal', ntot[run][1], flush=True)
    cols = ['run', 'i', 'read', 'q'] + [f'{f}@{t}' for t in PRE for f in ('sd', 'mad_diff', 'frac_hi')]
    T = pd.DataFrame(rows, columns=cols); T.to_csv(cache, sep='\t', index=False)
F = ('sd', 'mad_diff', 'frac_hi')
for t in PRE:
    for f in F:
        c = f'{f}@{t}'; T[c + '_rel'] = T[c] / T.groupby('run')[c].transform(lambda s: s.iloc[:500].median())
for qc in (8, 10, 12):
    T['low'] = (T.q < qc).astype(int)
    print(f'\n===== low = mean Q < {qc}  (same {len(T)} reads for every row; low share: ' + ', '.join(f'{r} {100*g.low.mean():.0f}%' for r, g in T.groupby('run')) + ') =====')
    print(f'{"prefix":<7}{"features":<16}{"mean AUC":>9}{"good kept":>11}{"low caught":>12}   (threshold from training runs; mean over held-out runs excluding the ladder)')
    for t in PRE:
        for name, feats in (('raw+relative', [f'{f}@{t}' for f in F] + [f'{f}@{t}_rel' for f in F]), ('relative', [f'{f}@{t}_rel' for f in F])):
            res = []
            for run in T.run.unique():
                if run in NOTRAIN: continue
                tr, te = T[(T.run != run) & ~T.run.isin(NOTRAIN)], T[T.run == run]
                if te.low.sum() < 20: continue
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[feats], tr.low)
                ptr = m.predict_proba(tr[feats])[:, 1]; pte = m.predict_proba(te[feats])[:, 1]; thr = np.percentile(ptr[tr.low.values == 0], 95)
                res.append((roc_auc_score(te.low, pte), (pte[te.low.values == 0] <= thr).mean(), (pte[te.low.values == 1] > thr).mean()))
            a = np.array(res); print(f'{t:<7}{name:<16}{a[:,0].mean():9.3f}{100*a[:,1].mean():10.0f}%{100*a[:,2].mean():11.0f}%   ({len(a)} runs)')
