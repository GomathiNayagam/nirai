"""Which signal features transfer across runs? Train on RDH275 (pass+fail), test on ladder pass/fail, KPC2 pass, RDH275 held-out pass.
Feature sets: all four, without median pA (sample-composition dependent), noise-only (mad_diff). usage: gate_ablation_cross_run.py <pass_dir> <fail_dir> <ladder_pass> <ladder_fail> <kpc2_pass> <rdh_heldout>"""
import sys, glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
from nirai.gate import feature_table, COLS
pa, fa, lp, lf, kp, ho = sys.argv[1:7]
T = lambda d, lab, n=None: pd.DataFrame(feature_table(sorted(glob.glob(d + '/*.pod5'))[:n], '0.4s', lab), columns=COLS).dropna(subset=['sd', 'mad_diff', 'med_pa', 'frac_hi'])
tr = pd.concat([T(pa, 1), T(fa, 0)]); LP, LF, K, HO = T(lp, 1, 4), T(lf, 0, 4), T(kp, 1, 3), T(ho, 1, 4)
for name, cols in (('all four', ['sd', 'mad_diff', 'med_pa', 'frac_hi']), ('without med_pa', ['sd', 'mad_diff', 'frac_hi']), ('mad_diff only', ['mad_diff']), ('sd + mad_diff', ['sd', 'mad_diff'])):
    m = H(max_iter=150, learning_rate=0.08, random_state=0).fit(tr[cols], tr.label)
    p = m.predict_proba(tr[cols])[:, 1]; thr = np.quantile(p[tr.label == 1], 0.05)            # in-sample threshold (95% pass kept on training run)
    sp, sf, sk, sh = (m.predict_proba(x[cols])[:, 1] for x in (LP, LF, K, HO))
    auc = roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sf))], np.r_[sp, sf]); t2 = np.quantile(sp, 0.05)
    print(f"{name:<15} ladder AUC {auc:.3f} | ladder at training threshold: pass kept {(sp>=thr).mean()*100:3.0f}% fail rej {(sf<thr).mean()*100:3.0f}% | re-set threshold: fail rej {(sf<t2).mean()*100:3.0f}% | pass kept KPC2 {(sk>=thr).mean()*100:3.0f}% RDH275-other {(sh>=thr).mean()*100:3.0f}%", flush=True)
