"""Cross-run test of a trained Nirai pass/fail gate. The model was trained on ONE run; here it scores other runs it never saw.
  ladder (PromethION R10.4.1, 1kb+ DNA ladder, pass and fail folders): AUC, and rates at (a) the threshold saved with the model, (b) a threshold re-set on the ladder
  pass-only/fail labels (separates 'ranking works' from 'calibration moved').
  pass-only sets (no fail labels): KPC2 and later RDH275 chunks -> only the share of pass reads kept at the saved threshold.
usage: gate_cross_run_eval.py <model.joblib> <ladder_pass_dir> <ladder_fail_dir> <pass_only_dir>:<label> ... [--files N]"""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from nirai.gate import load_model, feature_table, COLS
from nirai.features import FEATURES
mp, lp, lf = sys.argv[1:4]; extra = [a for a in sys.argv[4:] if ':' in a]; NF = int(sys.argv[sys.argv.index('--files') + 1]) if '--files' in sys.argv else 4
m = load_model(mp); thr = m['threshold']; print(f"model: prefix {m['prefix']}, threshold {thr:.3f}, trained on {m['cv']['n_pass']} pass / {m['cv']['n_fail']} fail reads from {m['cv']['n_files']} files; its CV: AUC {m['cv']['auc']:.3f}")
pick = lambda d: sorted(glob.glob(d + '/*.pod5'))[:NF]
def table(files, lab): return pd.DataFrame(feature_table(files, m['prefix'], lab), columns=COLS)
def score(d):
    ok = d.dropna(subset=list(FEATURES)); return ok, m['model'].predict_proba(ok[list(FEATURES)])[:, 1]
tp, tf = table(pick(lp), 1), table(pick(lf), 0)
print(f"\nladder: {len(tp)} pass reads ({tp.file.nunique()} files), {len(tf)} fail reads ({tf.file.nunique()} files); median length pass {tp.n_samples.median():.0f} fail {tf.n_samples.median():.0f} samples")
for nm, t in (('pass', tp), ('fail', tf)): print(f"  reads with >= prefix of signal ({nm}): {t[list(FEATURES)].notna().all(axis=1).mean()*100:.0f}%")
op, sp = score(tp); of, sf = score(tf); y = np.r_[np.ones(len(sp)), np.zeros(len(sf))]; s = np.r_[sp, sf]
print(f"  AUC pass vs fail: {roc_auc_score(y, s):.3f}")
print(f"  (a) saved threshold {thr:.3f}: pass kept {(sp >= thr).mean()*100:.0f}%, fail rejected {(sf < thr).mean()*100:.0f}%")
t2 = float(np.quantile(sp, 0.05)); print(f"  (b) threshold re-set on ladder labels for 95% pass kept ({t2:.3f}): fail rejected {(sf < t2).mean()*100:.0f}%")
for a in extra:
    d, lab = a.rsplit(':', 1); t = table(pick(d), 1); o, sc = score(t)
    print(f"\n{lab} (pass-only, {len(o)} reads with enough signal): pass kept at saved threshold {(sc >= thr).mean()*100:.0f}%  (expected ~95% if the model transfers)")
