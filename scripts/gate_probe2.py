"""Probe part 2 (inference only, shipped model): (A) which signal FREQUENCIES separate bad from good reads, (B) what the model MISSES / false-alarms on.
Data as in gate_probe.py: HG002 PromethION (4 runs x 8000 reads, unseen by the shipped model, label = pass/fail folder) and melb2 (in the model's training data, label = hac Q<10).
(A) power of the pA signal (median removed) in 9 frequency bands from 1 Hz to 2.5 kHz (5 kHz sampling): AUC of each band alone, median power for bad vs good, boosted trees on the bands vs on the 9 simple measures from gate_probe.py vs the CNN.
(B) among BAD reads: the 25% the model scores highest ('caught') vs the 25% it scores lowest ('missed'); among GOOD reads: the 10% scored highest ('false alarms') vs the rest.   usage: gate_probe2.py -> results/nirai_gate_probe2_spectrum_misses.txt"""
import os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
REPO = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]; sys.path.insert(0, REPO)
from nirai.gate_cnn import load_cnn_gate, normalize, scalars, relative
torch.set_num_threads(2); W = os.path.expanduser('~/rawsamble_work/'); V = W + 'varied/HG002_ALL/'; rng = np.random.default_rng(0); OUT = open(REPO + '/results/nirai_gate_probe2_spectrum_misses.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
g = load_cnn_gate(REPO + '/models/nirai_gate_cnn_1.0s.pt')
RUNS = {'0424': ('PAO89685', '20230424_1302_3H_PAO89685_2264ba8c'), '0428': ('PAO89685', '20230428_1310_3H_PAO89685_c9d0d53f'), '0429': ('PAO83395', '20230429_1600_2E_PAO83395_124388f5'), '0503': ('PAO83395', '20230503_1206_2E_PAO83395_ba06e1bb')}
BANDS = [(1, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 2500)]
def bands(raw):
    x = raw - np.median(raw, 1, keepdims=True); F = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2; fr = np.fft.rfftfreq(x.shape[1], 1 / 5000); tot = F[:, 1:].sum(1) + 1e-9
    return np.column_stack([np.log10(F[:, (fr >= a) & (fr < b)].sum(1) / tot + 1e-9) for a, b in BANDS]), np.log10(tot)       # log10 share of total power per band, log10 total power
def measures(raw):
    n = len(raw); med = np.median(raw, 1); sd = raw.std(1); d = np.diff(raw, axis=1); mad = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1); blk = np.median(raw.reshape(n, 50, 100), 2)
    return np.column_stack([sd, mad, (np.abs(raw - med[:, None]) > 4 * sd[:, None]).mean(1), med, blk.std(1), np.percentile(raw, 99, 1) - np.percentile(raw, 1, 1)])
MN = ['noise_sd', 'jitter_MAD', 'outlier_frac', 'median_pA', 'level_variability', 'range']
def plow(X, S):
    SC = np.hstack([S, np.tile(np.eye(3, dtype=np.float32)[0], (len(S), 1))]); o = []
    with torch.no_grad():
        for i in range(0, len(X), 2048): o.append(g.net(torch.from_numpy(X[i:i+2048])[:, None], torch.from_numpy(SC[i:i+2048])).numpy())
    return np.concatenate(o)
def inp(raw, ref): return normalize(raw).astype(np.float32), ((relative(scalars(raw), ref) - g.mean) / g.std).astype(np.float32)
def study(name, raw, y, lo, tr, te, lab):
    B, T = bands(raw); M = measures(raw); P(f'\n######## {name}: {len(y)} reads, bad {100 * y.mean():.1f}%; CNN AUC {roc_auc_score(y, lo):.3f}')
    P('\n== A. frequency bands (share of total power, log10): AUC alone (bad = positive), median for bad vs good ==\n  band (Hz)        AUC    median bad   median good')
    for j, (a, b) in enumerate(BANDS): P(f'  {a:>5}-{b:<5}    {roc_auc_score(y, B[:, j]):.3f}   {np.median(B[y == 1, j]):9.3f}   {np.median(B[y == 0, j]):9.3f}')
    P(f'  total power      {roc_auc_score(y, T):.3f}   {np.median(T[y == 1]):9.3f}   {np.median(T[y == 0]):9.3f}')
    def gb(F): return roc_auc_score(y[te], HistGradientBoostingClassifier(max_iter=200).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1])
    P(f'  trees, train {lab[0]} -> test {lab[1]}:  6 simple measures {gb(M):.3f} | 9 bands + total {gb(np.column_stack([B, T])):.3f} | both {gb(np.column_stack([M, B, T])):.3f} | CNN {roc_auc_score(y[te], lo[te]):.3f}')
    bad = np.where(y == 1)[0]; good = np.where(y == 0)[0]; cb = bad[lo[bad] >= np.percentile(lo[bad], 75)]; mb = bad[lo[bad] <= np.percentile(lo[bad], 25)]; fa = good[lo[good] >= np.percentile(lo[good], 90)]; rg = good[lo[good] < np.percentile(lo[good], 90)]
    P(f'\n== B. misses: BAD reads the model scores highest (caught, n {len(cb)}) vs lowest (missed, n {len(mb)}); GOOD reads scored highest 10% (false alarms, n {len(fa)}) vs rest (n {len(rg)}) ==')
    P(f'  {"measure":<22} {"caught":>9} {"missed":>9} {"false-alarm":>12} {"other good":>11}')
    for j, nme in enumerate(MN): P(f'  {nme:<22} {np.median(M[cb, j]):9.3f} {np.median(M[mb, j]):9.3f} {np.median(M[fa, j]):12.3f} {np.median(M[rg, j]):11.3f}')
    for j, (a, b) in enumerate(BANDS): P(f'  band {a}-{b} Hz (log share) {np.median(B[cb, j]):9.3f} {np.median(B[mb, j]):9.3f} {np.median(B[fa, j]):12.3f} {np.median(B[rg, j]):11.3f}')
    P(f'  mean CNN logit(P low): caught {lo[cb].mean():+.2f} missed {lo[mb].mean():+.2f} false-alarm {lo[fa].mean():+.2f} other good {lo[rg].mean():+.2f}')
raws, ys, fcs, los = [], [], [], []
for k, (fc, d) in RUNS.items():
    p = np.load(V + d + '/prefix5000.npz'); i = np.sort(rng.choice(len(p['FAIL']), 8000, replace=False)); raw = p['X'][i].astype(np.float32); ref = np.median(scalars(p['X'][rng.choice(len(p['FAIL']), 500, replace=False)].astype(np.float32)), axis=0)
    X, S = inp(raw, ref); raws.append(raw); ys.append(p['FAIL'][i]); fcs += [fc] * len(i); los.append(plow(X, S))
raw = np.concatenate(raws); y = np.concatenate(ys); fc = np.array(fcs); lo = np.concatenate(los)
study('HG002 PromethION (unseen by the model)', raw, y, lo, np.where(fc == 'PAO89685')[0], np.where(fc == 'PAO83395')[0], ('PAO89685', 'PAO83395'))
m5 = np.load(W + 'melb2/gate_eval/lc/melb2_prefix5000.npz', allow_pickle=True); Q = m5['QM'].astype(float); ok = ~np.isnan(Q); raw = m5['XM'][ok].astype(np.float32); y = (Q[ok] < 10).astype(int); o = np.argsort(m5['IM'][ok])
X, S = inp(raw, np.median(scalars(raw[o[:500]]), axis=0)); lo = plow(X, S); h = len(y) // 2
study('melb2 (IN the model training data)', raw, y, lo, o[:h], o[h:], ('first half', 'second half'))
