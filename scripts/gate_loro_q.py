"""Leave-one-run-out test of the signal-only gate with labels defined the SAME way in every run: label 'low' = mean read Q (from our own hac basecalls) < QCUT.
Features come from the first 0.4 s only. 'rel' features are divided by the median over the first 500 reads of the run (causal, usable live).
usage: gate_loro_q.py OUTDIR   (env QCUTS=8,10,12)"""
import os, sys, glob, numpy as np, pandas as pd, pod5
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.features import PREFIXES, prefix_features, read_pa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
W = os.path.expanduser('~/rawsamble_work/'); out = sys.argv[1]; os.makedirs(out, exist_ok=True)
RUNS = {  # name: (pod5 files, fastq files)
 'RDH275':      ([W+'rdh275/stream/first8000.pod5'], [W+'rdh275/stream/first8000_hac.fastq']),
 'KPC2_FAX76295': ([W+'kpc2_run/stream/first8000.pod5'], [W+'kpc2_run/stream/first8000_hac.fastq']),
 'KPC2_FAV69578': ([W+'kpc2_fc/FAV69578/stream/first8000.pod5'], [W+'kpc2_fc/FAV69578/stream/first8000_hac.fastq']),
 'KPC2_FAX39616': ([W+'kpc2_fc/FAX39616/stream/first8000.pod5'], [W+'kpc2_fc/FAX39616/stream/first8000_hac.fastq']),
 'KPC2_FAX39617': ([W+'kpc2_fc/FAX39617/stream/first8000.pod5'], [W+'kpc2_fc/FAX39617/stream/first8000_hac.fastq']),
 'LADDER_Oct':  ([W+'ladder/stream/part_00.pod5', W+'ladder/stream/part_01.pod5'], [W+'ladder/stream/part_00.fastq', W+'ladder/stream/part_01.fastq']),
 'ZYMO':        ([W+'zymo/stream/s3000.pod5'], [W+'zymo/stream/s3000_hac.fastq']),
}
def readq(fqs):
    q = {}
    for fq in fqs:
        with open(fq) as f:
            while True:
                h = f.readline()
                if not h: break
                f.readline(); f.readline(); a = np.frombuffer(f.readline().strip().encode(), np.uint8).astype(float) - 33
                if len(a): q[h[1:].split()[0]] = (-10 * np.log10(np.mean(10 ** (-a / 10))), len(a))
    return q
cache = out + '/features.tsv'
if os.path.exists(cache): T = pd.read_csv(cache, sep='\t')
else:
    rows = []; npre = PREFIXES['0.4s']
    for run, (ps, fqs) in RUNS.items():
        q = readq(fqs); k = 0
        for p in ps:
            with pod5.Reader(p) as r:
                for rec in r.reads():
                    rid = str(rec.read_id)
                    if rid not in q: continue
                    ft = prefix_features(read_pa(rec), npre)
                    if ft is None: continue
                    rows.append([run, k, rid, q[rid][0], q[rid][1]] + list(ft)); k += 1
        print(run, k, 'reads', flush=True)
    T = pd.DataFrame(rows, columns=['run', 'i', 'read', 'q', 'len', 'sd', 'mad_diff', 'frac_hi']); T.to_csv(cache, sep='\t', index=False)
F = ['sd', 'mad_diff', 'frac_hi']
NOTRAIN = [x for x in os.environ.get('NOTRAIN', '').split(',') if x]   # runs used only as held-out tests, never for training or threshold setting
for f in F:  # run-relative (first 500 reads of each run, in order)
    T[f + '_rel'] = T[f] / T.groupby('run').apply(lambda g: g.sort_values('i')[f].head(500).median()).reindex(T.run).values
print('\nrun summary (hac mean Q):'); print(T.groupby('run').q.describe()[['count', '25%', '50%', '75%']].round(1).to_string())
for qc in [int(x) for x in os.environ.get('QCUTS', '8,10,12').split(',')]:
    T['low'] = (T.q < qc).astype(int)
    print(f'\n===== "low quality" = mean Q < {qc} =====')
    print('share low per run: ' + ', '.join(f'{r} {100*g.low.mean():.0f}%' for r, g in T.groupby('run')))
    for name, feats in (('raw features', F), ('run-relative features', [f + '_rel' for f in F]), ('raw + relative', F + [f + '_rel' for f in F])):
        print(f'\n  -- {name} --\n  held-out run        n   low%   AUC   | threshold from TRAIN runs: good kept / low caught | threshold set on held-out run: low caught')
        res = []
        for run in T.run.unique():
            tr, te = T[(T.run != run) & ~T.run.isin(NOTRAIN)], T[T.run == run]
            if te.low.nunique() < 2 or tr.low.sum() < 50: continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[feats], tr.low)   # predicts P(low)
            ptr = m.predict_proba(tr[feats])[:, 1]; pte = m.predict_proba(te[feats])[:, 1]
            thr = np.percentile(ptr[tr.low.values == 0], 95)          # skip if P(low) > thr: keeps 95% of train good reads
            kept = (pte[te.low.values == 0] <= thr).mean(); caught = (pte[te.low.values == 1] > thr).mean()
            t2 = np.percentile(pte[te.low.values == 0], 95); c2 = (pte[te.low.values == 1] > t2).mean()
            auc = roc_auc_score(te.low, pte); res.append((auc, kept, caught, c2))
            print(f'  {run:<18}{len(te):6d} {100*te.low.mean():5.1f}% {auc:5.2f}   |   {100*kept:5.0f}% / {100*caught:5.0f}%   |   {100*c2:5.0f}%')
        a = np.array(res); print(f'  mean over runs: AUC {a[:,0].mean():.2f}, good kept {100*a[:,1].mean():.0f}%, low caught {100*a[:,2].mean():.0f}% (own-threshold {100*a[:,3].mean():.0f}%)')
