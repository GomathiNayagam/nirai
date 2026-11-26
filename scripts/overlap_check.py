"""Sanity check of the signal-position labels: do signal chunks of two reads at the SAME reference position look alike, and more than chunks of different positions?  usage: overlap_check.py"""
import os, pickle, numpy as np, pod5, collections
W = os.path.expanduser('~/rawsamble_work/melb2/'); lab = pickle.load(open(W + 'moves/labels.pkl', 'rb')); rng = np.random.default_rng(0)
ids = sorted(lab)[:12000]; cover = collections.defaultdict(list)
for r in ids:
    v = lab[r]; ok = v['base_ref'] >= 0; a, b = v['base_ref'][ok].min(), v['base_ref'][ok].max()
    for bb in range(a // 500, b // 500 + 1): cover[(v['chrom'], v['strand'], bb)].append(r)
pairs = []
for k, rs in cover.items():
    if len(rs) >= 2 and len(pairs) < 3000:
        a, b = rng.choice(len(rs), 2, replace=False); pairs.append((rs[a], rs[b], k[0], k[1], k[2] * 500 + 250))
need = {x for p in pairs for x in p[:2]}; sig = {}
with pod5.Reader(W + 'fc50k/stream/sorted.pod5') as rd:
    for rec in rd.reads(selection=list(need)):
        x = rec.signal.astype(np.float32); sig[str(rec.read_id)] = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
def chunk(r, pos, half=1000):
    v = lab[r]; i = np.flatnonzero(v['base_ref'] == pos)
    if len(i) == 0: j = np.flatnonzero(v['base_ref'] >= 0); i = [j[np.argmin(np.abs(v['base_ref'][j] - pos))]]
    c = int(v['base_sample'][i[0]]); s = sig[r]
    if c - half < 0 or c + half > len(s): return None
    x = np.clip(s[c - half:c + half], -4, 4).reshape(-1, 10).mean(1)
    return (x - x.mean()) / (x.std() + 1e-6)
def best_corr(a, b, maxs=8):  # allow small slip of up to maxs smoothed samples (80 samples ~ 7 bases)
    return max(np.mean(a[max(0, s):len(a) + min(0, s)] * b[max(0, -s):len(b) - max(0, s)]) for s in range(-maxs, maxs + 1))
same, diff, shifts = [], [], []
for ra, rb, ch, st, p in pairs:
    va, vb = lab[ra], lab[rb]; pa = p if st == '+' else p; a = chunk(ra, p); b = chunk(rb, p)
    if a is None or b is None: continue
    same.append(best_corr(a, b)); other = p + int(rng.choice([-1, 1]) * rng.integers(400, 1500)); b2 = chunk(rb, other)
    if b2 is not None: diff.append(best_corr(a, b2))
same, diff = np.array(same), np.array(diff)
print(f'pairs {len(same)}; chunk = 2000 samples (~160 bases), smoothed x10, best correlation over +-80-sample slip')
print(f'same reference position: mean {same.mean():.3f}, median {np.median(same):.3f}, 90th pct {np.percentile(same, 90):.3f}')
print(f'different position (400-1500 bases away): mean {diff.mean():.3f}, median {np.median(diff):.3f}, 90th pct {np.percentile(diff, 90):.3f}')
from sklearn.metrics import roc_auc_score
print('AUC same vs different by raw correlation:', round(roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(diff))], np.r_[same, diff]), 3))
