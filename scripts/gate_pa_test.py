"""Adds mean / median pA of the prefix to the gate features and re-runs leave-one-run-out (ladder never used for training).
Raw pA depends on the sample and run; 'rel' = value minus the median over the run's first 500 reads (live-usable).   usage: gate_pa_test.py OUTDIR"""
import os, sys, numpy as np, pandas as pd, pod5
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0]); sys.path.insert(0, os.path.dirname(__file__))
from nirai.features import read_pa, signal_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import importlib.util
src = open(os.path.join(os.path.dirname(__file__), 'gate_prefix_sweep.py')).read().split("cache = out")[0].split("W = os.path")[1]
ns = {'os': os, 'np': np, 'pd': pd, 'pod5': pod5, 'sys': sys}; exec("W = os.path" + src.replace("out = sys.argv[1]; os.makedirs(out, exist_ok=True)", "out = sys.argv[1]"), ns)
RUNS, readq, NOTRAIN = ns['RUNS'], ns['readq'], ns['NOTRAIN']; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
PRE = {'0.4s': 2000, '1.0s': 5000}; cache = out + '/pa_features.tsv'
if os.path.exists(cache): T = pd.read_csv(cache, sep='\t')
else:
    rows = []
    for run, (ps, fqs) in RUNS.items():
        q = readq(fqs); k = 0
        for p in ps:
            with pod5.Reader(p) as r:
                for rec in r.reads():
                    rid = str(rec.read_id)
                    if rid not in q: continue
                    x = read_pa(rec)
                    if len(x) < 5000: continue
                    row = [run, k, rid, q[rid]]
                    for n in PRE.values(): row += signal_features(x[:n]) + [float(np.median(x[:n])), float(x[:n].mean())]
                    rows.append(row); k += 1
        print(run, k, flush=True)
    T = pd.DataFrame(rows, columns=['run', 'i', 'read', 'q'] + [f'{f}@{t}' for t in PRE for f in ('sd', 'mad_diff', 'frac_hi', 'med_pa', 'mean_pa')]); T.to_csv(cache, sep='\t', index=False)
B = ('sd', 'mad_diff', 'frac_hi')
for t in PRE:
    for f in B + ('med_pa', 'mean_pa'):
        c = f'{f}@{t}'; med = T.groupby('run')[c].transform(lambda s: s.iloc[:500].median())
        T[c + '_rel'] = T[c] / med if f in B else T[c] - med
for qc in (10, 12):
    T['low'] = (T.q < qc).astype(int); print(f'\n===== low = mean Q < {qc} =====')
    print(f'{"prefix":<6}{"features":<34}{"mean AUC":>9}{"good kept":>11}{"low caught":>12}   (mean over held-out runs excluding the ladder)')
    for t in PRE:
        base = [f'{f}@{t}' for f in B] + [f'{f}@{t}_rel' for f in B]
        sets = {'base (noise, raw+rel)': base, '+ median pA raw': base + [f'med_pa@{t}'], '+ median pA relative': base + [f'med_pa@{t}_rel'],
                '+ mean pA relative': base + [f'mean_pa@{t}_rel'], '+ median & mean pA, raw+rel': base + [f'{a}@{t}{b}' for a in ('med_pa', 'mean_pa') for b in ('', '_rel')]}
        for name, feats in sets.items():
            res = []
            for run in T.run.unique():
                if run in NOTRAIN: continue
                tr, te = T[(T.run != run) & ~T.run.isin(NOTRAIN)], T[T.run == run]
                if te.low.sum() < 20: continue
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[feats], tr.low)
                ptr = m.predict_proba(tr[feats])[:, 1]; pte = m.predict_proba(te[feats])[:, 1]; thr = np.percentile(ptr[tr.low.values == 0], 95)
                res.append((roc_auc_score(te.low, pte), (pte[te.low.values == 0] <= thr).mean(), (pte[te.low.values == 1] > thr).mean()))
            a = np.array(res); print(f'{t:<6}{name:<34}{a[:,0].mean():9.3f}{100*a[:,1].mean():10.0f}%{100*a[:,2].mean():11.0f}%   ({len(a)} runs)', flush=True)
