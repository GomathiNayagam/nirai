"""Learning curve: does more training data (more runs) and a longer window (0.4 s vs 1.0 s) improve the CNN gate on a held-out run (melb2)?
Train on the first k runs of a fixed order (+ extra RDH275 fail/pass reads when RDH275 is in), test on melb2 (never in training; public-hac-call labels, Q<10 = low, no-call reads excluded).
Test set = melb2 reads with >= 1.0 s of signal, the same reads for both windows.   usage: gate_learning_curve.py   (results appended to melb2/gate_eval/lc/lc_results.jsonl)"""
import os, sys, json, time, numpy as np, pandas as pd, pod5, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.features import read_pa, signal_features
from sklearn.metrics import roc_auc_score
torch.set_num_threads(8)
W = os.path.expanduser('~/rawsamble_work/'); D = W + 'gate_loro/'; MB = W + 'melb2/fc50k/stream/'; OUT = W + 'melb2/gate_eval/lc/'; QC = 10
z = np.load(D + 'kit_data.npz', allow_pickle=True); XR, Q, R, I = z['XR'], z['Q'], z['R'], z['I']
e = np.load(D + 'failaug/extra.npz', allow_pickle=True); XE, QE = e['XE'], e['QE']
cache = OUT + 'melb2_prefix5000.npz'
if not os.path.exists(cache):
    qmap = {}
    with open(MB + 'first50000_hac.fastq') as f:
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().strip(); f.readline(); ql = f.readline().strip()
            if len(s) == len(ql) and ql: a = np.frombuffer(ql.encode(), np.uint8).astype(float) - 33; qmap[h[1:].split()[0]] = -10 * np.log10(np.mean(10 ** (-a / 10)))
    XM, QM, IM, RID = [], [], [], []
    with pod5.Reader(MB + 'sorted.pod5') as r:
        for k, rec in enumerate(r.reads()):
            x = read_pa(rec)
            if len(x) < 5000: continue
            rid = str(rec.read_id); XM.append(x[:5000].astype(np.float16)); QM.append(qmap.get(rid, np.nan)); IM.append(k); RID.append(rid)
    np.savez(cache, XM=np.stack(XM), QM=np.array(QM), IM=np.array(IM), RID=np.array(RID))
m5 = np.load(cache, allow_pickle=True); XM, QM, IM = m5['XM'], m5['QM'], m5['IM']
XA = np.concatenate([XR, XE, XM]); QA = np.concatenate([Q, QE, QM]); n0, ne = len(Q), len(QE)
RA = np.concatenate([R, np.array(['RDH275'] * ne), np.array(['MELB2'] * len(QM))]); IA = np.concatenate([I, 100000 + np.arange(ne), IM]); del XR, XE
KIT = {'RDH275': 0, 'MELB2': 0, 'KPC2_FAX76295': 0, 'KPC2_FAX39616': 0, 'KPC2_FAX39617': 0, 'ECOLI_ATCC25922': 0, 'KPC2_FAV69578': 1, 'ZYMO': 2, 'LADDER_Oct': 2}
have = ~np.isnan(QA); y = np.where(have, QA < QC, True).astype(np.float32); mel = np.where(RA == 'MELB2')[0]; te = mel[have[mel]]
ORDERS = {'A': ['KPC2_FAX39616', 'ECOLI_ATCC25922', 'RDH275', 'KPC2_FAX76295', 'KPC2_FAX39617', 'KPC2_FAV69578', 'ZYMO'], 'B': ['RDH275', 'KPC2_FAX76295', 'ECOLI_ATCC25922', 'KPC2_FAX39617', 'KPC2_FAX39616', 'KPC2_FAV69578', 'ZYMO']}
def prep(NP):
    raw = XA[:, :NP].astype(np.float32); med = np.median(raw, 1, keepdims=True); mad = np.median(np.abs(raw - med), 1, keepdims=True) + 1e-3
    X = np.clip((raw - med) / (1.4826 * mad), -6, 6).astype(np.float16)
    S = np.array([signal_features(r) + [float(np.median(r))] for r in raw], np.float32); S2 = np.zeros_like(S)
    for run in np.unique(RA):
        m = RA == run; ref = np.median(S[m][np.argsort(IA[m])[:500]], axis=0); S2[m, :3] = S[m, :3] / (ref[:3] + 1e-9); S2[m, 3] = (S[m, 3] - ref[3]) / 10
    S2 = np.hstack([S2, np.log(S[:, :3] + 1e-6)]); S2 = (S2 - S2.mean(0)) / (S2.std(0) + 1e-9)
    return X, np.hstack([S2, np.eye(3, dtype=np.float32)[[KIT[r] for r in RA]]])
class Net(nn.Module):
    def __init__(s, ns):
        super().__init__(); c = lambda i, o, k, st: nn.Sequential(nn.Conv1d(i, o, k, st, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        s.conv = nn.Sequential(c(1, 16, 9, 2), c(16, 32, 9, 4), c(32, 48, 9, 4), c(48, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64 + ns, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, sc): return s.head(torch.cat([s.conv(x), sc], 1)).squeeze(1)
def predict(m, X, SC, idx):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024): b = idx[i:i+1024]; o.append(torch.sigmoid(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b]))).numpy())
    return np.concatenate(o)
def fit(X, SC, tri, seed, epochs=6):
    torch.manual_seed(seed); np.random.seed(seed); m = Net(SC.shape[1]); pw = min((1 - y[tri].mean()) / y[tri].mean(), 20.0)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; loss = lf(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
    return m
def metrics(p, idx):
    yy = y[idx]; g = p[yy == 0]; l = p[yy == 1]; o = {'n': len(idx), 'low': int(yy.sum()), 'auc': float(roc_auc_score(yy, p))}
    for gk in (98, 99.8): thr = np.percentile(g, gk); o[f'catch@{gk}'] = float(100 * (l > thr).mean())      # own-run threshold: ranking quality only
    return o
res = open(OUT + 'lc_results.jsonl', 'a'); done = set()
if os.path.exists(OUT + 'lc_results.jsonl'):
    for l in open(OUT + 'lc_results.jsonl'): d = json.loads(l); done.add((d['window'], d['order'], d['k'], d['seed']))
STEPS = (1, 2, 3, 5, 7)
for NP in (2000, 5000):
    t0 = time.time(); X, SC = prep(NP); print(f'window {NP/5000:.1f}s prepared in {time.time()-t0:.0f}s', flush=True)
    te_ok = te   # all melb2 reads in the cache have >= 5000 samples, so the test set is the same for both windows
    for seed in (0, 1):
        for oname in ('A', 'B'):
            for k in STEPS:
                if (NP, oname, k, seed) in done: continue
                runs = ORDERS[oname][:k]; tri = np.where(np.isin(RA, runs))[0]; t0 = time.time(); m = fit(X, SC, tri, seed); p = predict(m, X, SC, te_ok); o = metrics(p, te_ok)
                o.update(window=NP, order=oname, k=k, seed=seed, runs=runs, n_train=int(len(tri)), low_train=int(y[tri].sum()), secs=round(time.time() - t0)); res.write(json.dumps(o) + '\n'); res.flush()
                print(f'{NP/5000:.1f}s order {oname} k={k} seed {seed} train {len(tri):6d} | AUC {o["auc"]:.3f} catch@98 {o["catch@98"]:.1f} catch@99.8 {o["catch@99.8"]:.1f} | {o["secs"]}s', flush=True)
