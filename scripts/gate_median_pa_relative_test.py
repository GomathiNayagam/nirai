"""Re-test median pA for the pass/fail gate in a run-relative form.
 abs   : median pA of the first 0.4 s (what failed to transfer)
 rel   : that median minus the running median of the previous 2000 reads of the SAME run (causal, no labels needed)
 relG  : minus the median over the whole run (upper bound; not causal)
Train on RDH275 (pass+fail files), test on ladder pass/fail, KPC2 pass-only, RDH275 other pass chunks.
usage: gate_median_pa_relative_test.py <rdh_pass> <rdh_fail> <lad_pass> <lad_fail> <kpc2_pass> <rdh_heldout>"""
import sys, glob, re, os, numpy as np, pandas as pd, pod5
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from nirai.features import signal_features, read_pa
N = 2000
def load(dirs_labels, nfiles=None):
    rows = []
    for d, lab in dirs_labels:
        for f in sorted(glob.glob(d + '/*.pod5'), key=lambda f: int(re.search(r'_(\d+)\.pod5', f).group(1)))[:nfiles]:
            num = int(re.search(r'_(\d+)\.pod5', f).group(1)); i = 0
            with pod5.Reader(f) as r:
                for rec in r.reads():
                    x = read_pa(rec)
                    if len(x) >= N: rows.append([num, i, lab] + signal_features(x[:N]) + [float(np.median(x[:N]))])
                    i += 1
    d = pd.DataFrame(rows, columns=['chunk', 'i', 'label', 'sd', 'mad_diff', 'frac_hi', 'med_pa']).sort_values(['chunk', 'i']).reset_index(drop=True)
    run = d.med_pa.rolling(2000, min_periods=200).median().shift(1)            # running median of previous reads (stream order)
    d['rel'] = d.med_pa - run.fillna(d.med_pa.iloc[:200].median()); d['relG'] = d.med_pa - d.med_pa.median(); return d
tr = load([(sys.argv[1], 1), (sys.argv[2], 0)]); L = load([(sys.argv[3], 1), (sys.argv[4], 0)], 4); K = load([(sys.argv[5], 1)]); HO = load([(sys.argv[6], 1)], 4)
print(f"train {len(tr)} reads (pass {tr.label.sum()}), ladder {len(L)}, KPC2 {len(K)}, RDH275-other {len(HO)}")
print("median pA of the first 0.4 s: RDH275 %.1f | ladder %.1f | KPC2 %.1f   (run-level shift is what breaks the absolute feature)" % (tr.med_pa.median(), L.med_pa.median(), K.med_pa.median()))
base = ['sd', 'mad_diff', 'frac_hi']
for name, cols in (('no median (current default)', base), ('+ abs median pA', base + ['med_pa']), ('+ run-relative median (causal)', base + ['rel']), ('+ whole-run-relative (upper bound)', base + ['relG'])):
    m = H(max_iter=150, learning_rate=0.08, random_state=0).fit(tr[cols], tr.label)
    p = m.predict_proba(tr[cols])[:, 1]; thr = np.quantile(p[tr.label == 1], 0.05)
    pl, pk, ph = (m.predict_proba(x[cols])[:, 1] for x in (L, K, HO)); y = L.label.values; auc = roc_auc_score(y, pl)
    sp, sf = pl[y == 1], pl[y == 0]; t2 = np.quantile(sp, 0.05)
    print(f"{name:<36} ladder AUC {auc:.3f} | at training threshold: pass kept {(sp>=thr).mean()*100:3.0f}% fail rej {(sf<thr).mean()*100:3.0f}% | re-set: fail rej {(sf<t2).mean()*100:3.0f}% | pass kept KPC2 {(pk>=thr).mean()*100:3.0f}% RDH275-other {(ph>=thr).mean()*100:3.0f}%", flush=True)
