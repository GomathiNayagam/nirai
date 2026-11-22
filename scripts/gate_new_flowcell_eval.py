"""Score pass/fail POD5 files of a new flow cell with a saved gate; no retraining, no tuning.
usage: gate_new_flowcell_eval.py MODEL DIR [reads_per_label=5000]"""
import sys, os, glob, random
import numpy as np, pandas as pd, pod5
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.gate import load_model
from nirai.features import PREFIXES, prefix_features, read_pa
from sklearn.metrics import roc_auc_score
m = load_model(sys.argv[1]); d = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 5000; npre = PREFIXES[m['prefix']]; rng = random.Random(0)
rows = []
for lab in ('pass', 'fail'):
    fs = sorted(glob.glob(f'{d}/**/*_{lab}_*.pod5', recursive=True)); fs = rng.sample(fs, min(len(fs), 6)); per = -(-n // len(fs))
    for f in fs:
        with pod5.Reader(f) as r:
            for i, rec in enumerate(r.reads()):
                if i >= per: break
                ft = prefix_features(read_pa(rec), npre)
                if ft is not None: rows.append([lab, os.path.basename(f)] + ft)
t = pd.DataFrame(rows, columns=['label', 'file'] + list(m['features'])); t['score'] = m['model'].predict_proba(t[m['features']])[:, 1]
y = (t.label == 'pass').astype(int); thr = m['threshold']
print(f"{d}: {(t.label=='pass').sum()} pass / {(t.label=='fail').sum()} fail reads ({t.file.nunique()} files)")
print(f"AUC pass vs fail: {roc_auc_score(y, t.score):.3f}")
print(f"(a) saved threshold {thr:.3f}: pass kept {100*(t[t.label=='pass'].score>=thr).mean():.0f}%, fail rejected {100*(t[t.label=='fail'].score<thr).mean():.0f}%")
t95 = np.percentile(t[t.label == 'pass'].score, 5)
print(f"(b) threshold re-set on this flow cell for 95% pass kept ({t95:.3f}): fail rejected {100*(t[t.label=='fail'].score<t95).mean():.0f}%")
print("per-file pass-kept / fail-rejected at saved threshold:"); print(t.assign(kept=t.score >= thr).groupby(['label', 'file']).kept.mean().round(2).to_string())
