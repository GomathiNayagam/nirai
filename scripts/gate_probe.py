"""What does the 1.0 s CNN gate look at?  Probes the SHIPPED model (models/nirai_gate_cnn_1.0s.pt: MinION, native kit, trained on 5 base runs + melb2 + 5 species, NEVER on HG002) on
  HG002 PromethION (4 runs, 8000 random reads each, label = pass/fail folder; the model never saw this data)   and   melb2 (all labelled reads, label = public hac Q<10; NOTE melb2 WAS in the model's training data, so read it as 'what it uses', not as a test).
1. part ablation (no retraining; set the input to its neutral value, so read as rough):  full | waveform neutral (flat at the median) | summary numbers neutral (training mean)
2. occlusion: flatten one 0.1 s slice (500 samples) of the waveform at a time; AUC and mean change in the model's P(low) for bad vs good reads.   Which part of the first second matters?
3. single signal measures: AUC of each simple measure alone, and Spearman correlation with the model score; a small boosted-tree model on these measures alone (train: other flow cell / first half of the run) vs the CNN.
4. what the most-flagged reads look like: median of each measure for the top 2% model scores vs the rest.
5. time: Spearman of model score with read start time, inside the fail class and inside the pass class (HG002).
usage: gate_probe.py   -> results/nirai_gate_probe_hg002_melb2.txt"""
import os, sys, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]); REPO = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]
from nirai.gate_cnn import load_cnn_gate, normalize, scalars, relative
torch.set_num_threads(2); W = os.path.expanduser('~/rawsamble_work/'); V = W + 'varied/HG002_ALL/'; rng = np.random.default_rng(0); OUT = open(REPO + '/results/nirai_gate_probe_hg002_melb2.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
g = load_cnn_gate(REPO + '/models/nirai_gate_cnn_1.0s.pt')
RUNS = {'0424': ('PAO89685', '20230424_1302_3H_PAO89685_2264ba8c'), '0428': ('PAO89685', '20230428_1310_3H_PAO89685_c9d0d53f'), '0429': ('PAO83395', '20230429_1600_2E_PAO83395_124388f5'), '0503': ('PAO83395', '20230503_1206_2E_PAO83395_ba06e1bb')}
def measures(raw):
    raw = np.asarray(raw, np.float32); n = len(raw); med = np.median(raw, 1); sd = raw.std(1); d = np.diff(raw, axis=1); mad = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1)
    z = (raw - med[:, None]) / (1.4826 * np.median(np.abs(raw - med[:, None]), 1, keepdims=True) + 1e-3); blk = np.median(raw.reshape(n, 50, 100), 2)
    return {'noise_sd (pA)': sd, 'noise_diff_MAD (pA)': mad, 'outlier_frac (>4sd)': (np.abs(raw - med[:, None]) > 4 * sd[:, None]).mean(1), 'median_level (pA)': med,
            'spike_frac (|z|>3)': (np.abs(z) > 3).mean(1), 'level_variability (sd of 100-sample medians)': blk.std(1), 'start_offset (first 0.2 s median - rest, pA)': np.median(raw[:, :1000], 1) - np.median(raw[:, 1000:], 1),
            'low_current_frac (<60% of median)': (raw < 0.6 * med[:, None]).mean(1), 'range_p99-p1 (pA)': np.percentile(raw, 99, 1) - np.percentile(raw, 1, 1)}
def inputs(raw, ref):
    X = normalize(raw).astype(np.float32); S = ((relative(scalars(raw), ref) - g.mean) / g.std).astype(np.float32); return X, S
def plow(X, S, kit=0, batch=2048):
    SC = np.hstack([S, np.tile(np.eye(3, dtype=np.float32)[kit], (len(S), 1))]); o = []
    with torch.no_grad():
        for i in range(0, len(X), batch): o.append(g.net(torch.from_numpy(X[i:i+batch])[:, None], torch.from_numpy(SC[i:i+batch])).numpy())
    return np.concatenate(o)            # logit of P(low)
def probe(name, raw, y, XS, st=None, fc=None, order=None):
    P(f'\n######## {name}: {len(y)} reads, bad/fail {100 * y.mean():.1f}%'); X, S = XS; lo = plow(X, S); base = roc_auc_score(y, lo)
    P(f'full model AUC {base:.3f}  (kit slot fed = native)')
    P('\n== 1. part ablation (AUC) ==')
    P(f'  waveform made flat (summary numbers only): {roc_auc_score(y, plow(np.zeros_like(X), S)):.3f}')
    P(f'  summary numbers set to training mean (waveform only): {roc_auc_score(y, plow(X, np.zeros_like(S))):.3f}')
    P('\n== 2. occlusion: flatten one 0.1 s slice at a time ==\n  slice(s)   AUC   mean change in logit P(low): bad reads | good reads')
    for k in range(10):
        Xo = X.copy(); Xo[:, k * 500:(k + 1) * 500] = 0; lo2 = plow(Xo, S); P(f'  {k * 0.1:.1f}-{(k + 1) * 0.1:.1f}   {roc_auc_score(y, lo2):.3f}   {np.mean(lo2[y == 1] - lo[y == 1]):+.2f} | {np.mean(lo2[y == 0] - lo[y == 0]):+.2f}')
    M = measures(raw); P('\n== 3. single measures: AUC alone (bad=positive; <0.5 means bad reads have LOWER values) and Spearman with model score ==')
    for k, v in M.items(): a = roc_auc_score(y, v); P(f'  {k:<52} AUC {a:.3f}   Spearman with model {spearmanr(v, lo)[0]:+.2f}')
    F = np.column_stack(list(M.values())); tr, te = order
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(F[tr], y[tr]); P(f'  boosted trees on these {F.shape[1]} measures alone (train {fc[0]}, test {fc[1]}): AUC {roc_auc_score(y[te], gb.predict_proba(F[te])[:, 1]):.3f}   vs CNN on the same test reads: {roc_auc_score(y[te], lo[te]):.3f}')
    thr = np.percentile(lo, 98); top = lo > thr; P(f'\n== 4. median of each measure: top 2% model scores (n {int(top.sum())}, {100 * y[top].mean():.0f}% bad) vs the other reads ==')
    for k, v in M.items(): P(f'  {k:<52} top2% {np.median(v[top]):9.3f}   others {np.median(v[~top]):9.3f}   bad {np.median(v[y == 1]):9.3f}   good {np.median(v[y == 0]):9.3f}')
    if st is not None:
        P('\n== 5. time: Spearman of model score with read start time =='); P(f'  inside bad/fail reads {spearmanr(st[y == 1], lo[y == 1])[0]:+.2f}   inside good/pass reads {spearmanr(st[y == 0], lo[y == 0])[0]:+.2f}   (run-wide {spearmanr(st, lo)[0]:+.2f})')
# ---- HG002
raws, ys, sts, fcs, refs = [], [], [], [], []
for k, (fc, d) in RUNS.items():
    p = np.load(V + d + '/prefix5000.npz'); i = np.sort(rng.choice(len(p['FAIL']), 8000, replace=False)); raw = p['X'][i].astype(np.float32); ref = np.median(scalars(p['X'][rng.choice(len(p['FAIL']), 500, replace=False)].astype(np.float32)), axis=0)
    raws.append(raw); ys.append(p['FAIL'][i]); sts.append(p['ST'][i]); fcs += [fc] * len(i); refs.append(ref)
# the run-relative reference is per run: build inputs run by run
raw = np.concatenate(raws); y = np.concatenate(ys); st = np.concatenate(sts); fc = np.array(fcs); XS = [inputs(r, ref) for r, ref in zip(raws, refs)]   # run-relative inputs are built run by run
XS = (np.concatenate([a for a, _ in XS]), np.concatenate([b for _, b in XS])); tr = np.where(fc == 'PAO89685')[0]; te = np.where(fc == 'PAO83395')[0]
probe('HG002 PromethION (4 runs, unseen by the model)', raw, y, XS, st=st, fc=('PAO89685', 'PAO83395'), order=(tr, te))
# ---- melb2
m5 = np.load(W + 'melb2/gate_eval/lc/melb2_prefix5000.npz', allow_pickle=True); Q = m5['QM'].astype(float); ok = ~np.isnan(Q); raw = m5['XM'][ok].astype(np.float32); y = (Q[ok] < 10).astype(int); I = m5['IM'][ok]; o = np.argsort(I)
ref = np.median(scalars(raw[o[:500]]), axis=0); h = len(y) // 2; tr = o[:h]; te = o[h:]
probe('melb2 (V. parahaemolyticus; IN the model training data)', raw, y, inputs(raw, ref), fc=('first half of run', 'second half'), order=(tr, te))
