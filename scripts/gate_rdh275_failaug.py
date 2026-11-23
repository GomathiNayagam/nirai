"""Does adding real RDH275 fail reads (hac-basecalled, Q<10 labels) to CNN training help? 1D CNN + scalars + kit input, first 0.4 s, ladder never used.
Extra RDH275 reads: fail folder reads (all 20,000 basecalled) + 4,000 random pass reads, none of which are in the 8,000-read RDH275 stream used before.
 T1  test = RDH275 stream (natural mix):   base = other runs only (RDH275 held out)   vs   aug = base + extra RDH275 reads (same run, different reads)
 T2  test = other runs (leave-one-run-out): base = all other runs (incl. RDH275 stream)  vs   aug = base + extra RDH275 reads
 T3  test = held-out RDH275 FAIL file chunk + pass reads: train on others + remaining extra reads (whole-file hold-out)
usage: gate_rdh275_failaug.py   (nice it; uses 6 threads)"""
import os, sys, numpy as np, pandas as pd, pod5, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.features import read_pa, signal_features
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6); torch.manual_seed(0); np.random.seed(0)
W = os.path.expanduser('~/rawsamble_work/'); D = W + 'gate_loro/'; M = W + 'rdh275/moredata/'; NP = 2000; N = 5000; QC = 10
z = np.load(D + 'kit_data.npz', allow_pickle=True); XR, Q, R, I = z['XR'], z['Q'], z['R'], z['I']
def qmean(fq, drop_last=False):
    q = {}; recs = []
    with open(fq) as f:
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().strip(); f.readline(); ql = f.readline().strip()
            recs.append((h[1:].split()[0], s, ql))
    if drop_last and recs: recs = recs[:-1]
    for rid, s, ql in recs:
        if len(s) == len(ql) and len(ql):
            a = np.frombuffer(ql.encode(), np.uint8).astype(float) - 33; q[rid] = -10 * np.log10(np.mean(10 ** (-a / 10)))
    return q
stream_ids = set(l[1:].split()[0] for i, l in enumerate(open(W + 'rdh275/stream/first8000_hac.fastq')) if i % 4 == 0)
R0 = set(l.strip() for l in open(M + 'fail_rest_0.ids'))
cache = D + 'failaug/extra.npz'
if os.path.exists(cache): e = np.load(cache, allow_pickle=True); XE, QE, SE = e['XE'], e['QE'], e['SE']
else:
    XE = []; QE = []; SE = []
    for src, pod, fqs, ids in (('fail', M + 'fail_all.pod5', [(M + 'fail_part0_done.fastq', True), (M + 'fail_rest_0.fastq', False), (M + 'fail_rest_1.fastq', False)], None), ('pass', M + 'pass_extra.pod5', [(M + 'pass_4000.fastq', False)], None)):
        q = {}
        for fq, dl in fqs:
            if os.path.exists(fq): q.update(qmean(fq, dl))
        q = {k: v for k, v in q.items() if k not in stream_ids}; print(src, 'basecalled reads', len(q), flush=True)
        with pod5.Reader(pod) as r:
            for rec in r.reads(selection=list(q), missing_ok=True):
                x = read_pa(rec)
                if len(x) < N: continue
                rid = str(rec.read_id); XE.append(x[:N].astype(np.float16)); QE.append(q[rid]); SE.append('fail_rest_0' if (src == 'fail' and rid in R0) else src)
    XE = np.stack(XE); QE = np.array(QE); SE = np.array(SE); np.savez(cache, XE=XE, QE=QE, SE=SE)
print('extra reads (>=1 s):', {s: int((SE == s).sum()) for s in np.unique(SE)}, 'low share:', {s: round(float((QE[SE == s] < QC).mean()), 3) for s in np.unique(SE)}, flush=True)
# combine; extra reads are RDH275 run (kit NBD) with run-relative reference taken from the stream reads
n0 = len(Q); XR = np.concatenate([XR, XE]); Q = np.concatenate([Q, QE]); R = np.concatenate([R, np.array(['RDH275'] * len(QE))]); I = np.concatenate([I, 100000 + np.arange(len(QE))])
SRC = np.concatenate([np.array(['base'] * n0), SE])
raw = XR[:, :NP].astype(np.float32); med = np.median(raw, 1, keepdims=True); mad = np.median(np.abs(raw - med), 1, keepdims=True) + 1e-3
X = np.clip((raw - med) / (1.4826 * mad), -6, 6).astype(np.float16)
S = np.array([signal_features(r) + [float(np.median(r))] for r in raw], np.float32); S2 = np.zeros_like(S)
for run in np.unique(R):
    m = R == run; ref = np.median(S[m][np.argsort(I[m])[:500]], axis=0); S2[m, :3] = S[m, :3] / (ref[:3] + 1e-9); S2[m, 3] = (S[m, 3] - ref[3]) / 10
S2 = np.hstack([S2, np.log(S[:, :3] + 1e-6)]); S2 = (S2 - S2.mean(0)) / (S2.std(0) + 1e-9)
KIT = {'RDH275': 0, 'KPC2_FAX76295': 0, 'KPC2_FAX39616': 0, 'KPC2_FAX39617': 0, 'ECOLI_ATCC25922': 0, 'KPC2_FAV69578': 1, 'ZYMO': 2, 'LADDER_Oct': 2}
KS = np.eye(3, dtype=np.float32)[[KIT[r] for r in R]]; SC = np.hstack([S2, KS]); y = (Q < QC).astype(np.float32)
class Net(nn.Module):
    def __init__(s, ns):
        super().__init__(); c = lambda i, o, k, st: nn.Sequential(nn.Conv1d(i, o, k, st, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        s.conv = nn.Sequential(c(1, 16, 9, 2), c(16, 32, 9, 4), c(32, 48, 9, 4), c(48, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64 + ns, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, sc): return s.head(torch.cat([s.conv(x), sc], 1)).squeeze(1)
def predict(m, idx):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024): b = idx[i:i+1024]; o.append(torch.sigmoid(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b]))).numpy())
    return np.concatenate(o)
def fit(tri, epochs=6):
    m = Net(SC.shape[1]); pw = min((1 - y[tri].mean()) / y[tri].mean(), 20.0); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; loss = lf(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
    return m
def report(tag, m, tri, te):
    ptr = predict(m, tri); pte = predict(m, te); a = roc_auc_score(y[te], pte); g = pte[y[te] == 0]; l = pte[y[te] == 1]
    out = f'{tag:<44}train {len(tri):6d} test {len(te):6d} (low {int(y[te].sum()):5d}) AUC {a:.3f}'
    for pc in (99, 98, 95):   # good-read-first thresholds: percentile of TRAINING good-read scores (carried over) | percentile of the run's own good-read scores (ceiling)
        thr = np.percentile(ptr[y[tri] == 0], pc); out += f' | {pc}%: carried good kept {100*(g<=thr).mean():3.0f}% low caught {100*(l>thr).mean():3.0f}%, own-run low caught {100*(l>np.percentile(g, pc)).mean():3.0f}%'
    print(out, flush=True)
notr = ~np.isin(R, ['LADDER_Oct']); stream = (R == 'RDH275') & (SRC == 'base'); extra = SRC != 'base'
print('\n== T1: test = RDH275 stream (natural mix) ==', flush=True)
te = np.where(stream)[0]
tri = np.where(notr & (R != 'RDH275'))[0]; m = fit(tri); report('base (RDH275 held out)', m, tri, te)
tri = np.where(notr & ((R != 'RDH275') | extra))[0]; m = fit(tri); report('aug  (+ extra RDH275 fail/pass reads)', m, tri, te)
print('\n== T3: whole-file hold-out: test = fail_rest_0 chunk + 1,000 pass reads; train = others + remaining extra ==', flush=True)
ho = np.where(SRC == 'fail_rest_0')[0]; pidx = np.where(SRC == 'pass')[0]; rp = np.random.RandomState(0).permutation(pidx); te = np.concatenate([ho, rp[:1000]])
tri = np.where(notr & ((R != 'RDH275') | (extra & ~np.isin(np.arange(len(y)), te))))[0]; m = fit(tri); report('aug, held-out fail chunk + 1000 pass', m, tri, te)
tri = np.where(notr & (R != 'RDH275'))[0]; m = fit(tri); report('base, same test', m, tri, te)
print('\n== T2: other runs held out ==', flush=True)
for run in ['KPC2_FAX39616', 'ECOLI_ATCC25922', 'KPC2_FAX76295']:
    te = np.where(R == run)[0]
    tri = np.where(notr & (R != run) & ~extra)[0]; m = fit(tri); report(f'base, {run} held out', m, tri, te)
    tri = np.where(notr & (R != run))[0]; m = fit(tri); report(f'aug,  {run} held out', m, tri, te)
