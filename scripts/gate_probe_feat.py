"""Jitter vs high-frequency share: is one better, and does it carry over?  Per-read features from the first 1.0 s of raw pA (no run reference needed):
  jitter = median |diff - median diff| (pA);  hf = log10 share of power at 1600-2500 Hz;  hf2 = log10 share at 800-2500 Hz.
Logistic regression (standardised) trained on ONE dataset, tested on the others; bad = positive.  Reported per test set: AUC, and with the threshold fixed on the TRAINING set at 98% of its good reads kept: % good flagged and % bad caught.
Train sets: melb2 (MinION, hac Q<10), HG002 flow cell PAO89685 (PromethION, folder label).  Test: everything else (5 species, melb2, both HG002 flow cells).   usage: gate_probe_feat.py -> results/nirai_gate_probe_features.txt"""
import os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
REPO = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]; W = os.path.expanduser('~/rawsamble_work/'); V = W + 'varied/'; OUT = open(REPO + '/results/nirai_gate_probe_features.txt', 'w'); rng = np.random.default_rng(0)
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
def feats(raw):
    return np.concatenate([feats1(raw[i:i + 4000]) for i in range(0, len(raw), 4000)])      # chunks: the FFT of a whole dataset does not fit in memory
def feats1(raw):
    raw = np.asarray(raw, np.float32); d = np.diff(raw, axis=1); jit = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1); x = raw - np.median(raw, 1, keepdims=True)
    F = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2; fr = np.fft.rfftfreq(x.shape[1], 1 / 5000); tot = F[:, 1:].sum(1) + 1e-9
    return np.column_stack([jit, np.log10(F[:, fr >= 1600].sum(1) / tot + 1e-9), np.log10(F[:, fr >= 800].sum(1) / tot + 1e-9)])
DS = {}
m5 = np.load(W + 'melb2/gate_eval/lc/melb2_prefix5000.npz', allow_pickle=True); Q = m5['QM'].astype(float); ok = ~np.isnan(Q); DS['melb2'] = (feats(m5['XM'][ok].astype(np.float32)), (Q[ok] < 10).astype(int))
for n in ('CJEJ_33560', 'LIVAN_19119', 'SAUR_BPH2947', 'MTB_AMtb1', 'SALM_10708'):
    p = np.load(V + n + '/prefix5000.npz', allow_pickle=True); Q = p['Q'].astype(float); ok = ~np.isnan(Q); DS[n] = (feats(p['X'][ok].astype(np.float32)), (Q[ok] < 10).astype(int))
for fc, runs in (('HG002_PAO89685', ('20230424_1302_3H_PAO89685_2264ba8c', '20230428_1310_3H_PAO89685_c9d0d53f')), ('HG002_PAO83395', ('20230429_1600_2E_PAO83395_124388f5', '20230503_1206_2E_PAO83395_ba06e1bb'))):
    Fs, ys = [], []
    for d in runs: p = np.load(V + 'HG002_ALL/' + d + '/prefix5000.npz'); i = rng.choice(len(p['FAIL']), 15000, replace=False); Fs.append(feats(p['X'][i].astype(np.float32))); ys.append(p['FAIL'][i])
    DS[fc] = (np.concatenate(Fs), np.concatenate(ys))
SETS = {'jitter only': [0], 'high-freq share only (1600-2500 Hz)': [1], 'high-freq share only (800-2500 Hz)': [2], 'jitter + hf 1600': [0, 1], 'jitter + hf 800': [0, 2]}
for tr in ('melb2', 'HG002_PAO89685'):
    P(f'\n######## trained on {tr} (n {len(DS[tr][1])}, bad {100 * DS[tr][1].mean():.1f}%); threshold fixed at 98% of its good reads kept')
    for name, cols in SETS.items():
        sc = StandardScaler().fit(DS[tr][0][:, cols]); lr = LogisticRegression(max_iter=1000, class_weight='balanced').fit(sc.transform(DS[tr][0][:, cols]), DS[tr][1]); f = lambda k: lr.decision_function(sc.transform(DS[k][0][:, cols]))
        thr = np.percentile(f(tr)[DS[tr][1] == 0], 98); P(f'  -- {name}   (weights {np.round(lr.coef_[0], 2)})')
        for te in DS:
            if te == tr: continue
            s = f(te); y = DS[te][1]; P(f'     test {te:<16} AUC {roc_auc_score(y, s):.3f}   good flagged {100 * (s[y == 0] > thr).mean():5.1f}%   bad caught {100 * (s[y == 1] > thr).mean():5.1f}%')
