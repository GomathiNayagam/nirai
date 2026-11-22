"""Kit-aware training study for the signal-only fail-likely gate (1D CNN + scalars), leave-one-run-out, ladder never used for training.
Strategies for each held-out run:  pooled = all other non-ladder runs;  samekit = only other runs with the same library kit;  pooled+kit = pooled with the kit as an input.
Looks at the first PREFIX samples only (2000 = 0.4 s, 3000 = 0.6 s).  low = mean Q (our hac basecalls; E. coli run: Q from its SRA basecalls) < QCUT.
usage: gate_kit_cnn.py OUTDIR PREFIX [QCUT=10]"""
import os, sys, random, glob, numpy as np, pandas as pd, pod5, torch, torch.nn as nn
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.features import read_pa, signal_features
from sklearn.metrics import roc_auc_score
torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
out = sys.argv[1]; NP = int(sys.argv[2]); QC = float(sys.argv[3]) if len(sys.argv) > 3 else 10; N = 5000
W = os.path.expanduser('~/rawsamble_work/')
ns = {}; src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate_prefix_sweep.py')).read().split("cache = out")[0].split("W = os.path")[1]
exec("import os,numpy as np\nW = os.path" + src.replace("out = sys.argv[1]; os.makedirs(out, exist_ok=True)", "out=''"), {'os': os, 'np': np, 'pd': pd, 'pod5': pod5, 'sys': sys}, ns)
RUNS, readq = ns['RUNS'], ns['readq']
KIT = {'RDH275': 'NBD', 'KPC2_FAX76295': 'NBD', 'KPC2_FAX39616': 'NBD', 'KPC2_FAX39617': 'NBD', 'ECOLI_ATCC25922': 'NBD', 'KPC2_FAV69578': 'RBK', 'ZYMO': 'LSK', 'LADDER_Oct': 'LSK'}
NOTRAIN = ['LADDER_Oct']
cache = out + '/kit_data.npz'
if os.path.exists(cache): z = np.load(cache, allow_pickle=True); XR, Q, R, I = z['XR'], z['Q'], z['R'], z['I']
else:
    XR = []; Q = []; R = []; I = []
    for run, (ps, fqs) in RUNS.items():
        q = readq(fqs); k = 0
        for p in ps:
            with pod5.Reader(p) as r:
                for rec in r.reads():
                    rid = str(rec.read_id)
                    if rid not in q: continue
                    x = read_pa(rec)
                    if len(x) < N: continue
                    XR.append(x[:N].astype(np.float16)); Q.append(q[rid]); R.append(run); I.append(k); k += 1
        print(run, k, flush=True)
    e = pd.read_csv(W + 'ecoli_q/ecoli_sra_q.tsv', sep='\t'); e = e.sample(n=min(30000, len(e)), random_state=0); base = os.path.expanduser('~/thusumbu/data/rawsamble_replication/d5_ecoli_r1041/pod5/'); k = 0
    for f, g in e.groupby('file'):
        qm = dict(zip(g.read, g.q))
        with pod5.Reader(base + f) as r:
            for rec in r.reads(selection=list(qm)):
                x = read_pa(rec)
                if len(x) < N: continue
                XR.append(x[:N].astype(np.float16)); Q.append(qm[str(rec.read_id)]); R.append('ECOLI_ATCC25922'); I.append(k); k += 1
    print('ECOLI_ATCC25922', k, flush=True)
    XR = np.stack(XR); Q = np.array(Q); R = np.array(R); I = np.array(I); np.savez(cache, XR=XR, Q=Q, R=R, I=I)
raw = XR[:, :NP].astype(np.float32)
med = np.median(raw, 1, keepdims=True); mad = np.median(np.abs(raw - med), 1, keepdims=True) + 1e-3
X = np.clip((raw - med) / (1.4826 * mad), -6, 6).astype(np.float16)
S = np.array([signal_features(r) + [float(np.median(r))] for r in raw], np.float32)
S2 = np.zeros_like(S)
for run in np.unique(R):
    m = R == run; ref = np.median(S[m][np.argsort(I[m])[:500]], axis=0)
    S2[m, :3] = S[m, :3] / (ref[:3] + 1e-9); S2[m, 3] = (S[m, 3] - ref[3]) / 10
S2 = np.hstack([S2, np.log(S[:, :3] + 1e-6)]); S2 = (S2 - S2.mean(0)) / (S2.std(0) + 1e-9)
KS = np.array([[KIT[r] == k for k in ('NBD', 'RBK', 'LSK')] for r in R], np.float32)
y = (Q < QC).astype(np.float32)
class Net(nn.Module):
    def __init__(s, ns):
        super().__init__(); c = lambda i, o, k, st: nn.Sequential(nn.Conv1d(i, o, k, st, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        s.conv = nn.Sequential(c(1, 16, 9, 2), c(16, 32, 9, 4), c(32, 48, 9, 4), c(48, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear(64 + ns, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, sc): return s.head(torch.cat([s.conv(x), sc], 1)).squeeze(1)
def predict(m, Sc, idx):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024): b = idx[i:i+1024]; o.append(torch.sigmoid(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(Sc[b]))).numpy())
    return np.concatenate(o)
def fit(tri, Sc, epochs=6):
    m = Net(Sc.shape[1]); pw = min((1 - y[tri].mean()) / y[tri].mean(), 20.0); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; loss = lf(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(Sc[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
    return m
print(f'prefix {NP} samples ({NP/5000:.1f} s); low = mean Q < {QC}; reads {len(y)}; low share: ' + ', '.join(f'{r} {100*y[R==r].mean():.1f}%' for r in np.unique(R)), flush=True)
SCK = np.hstack([S2, KS])
print(f'\n{"held-out run (kit)":<24}{"strategy":<12}{"train n":>8}{"AUC":>7}{"good kept":>11}{"low caught":>12}{"low caught @95% kept on this run":>34}', flush=True)
res = {}
for run in ['RDH275', 'KPC2_FAX39616', 'KPC2_FAX39617', 'KPC2_FAX76295', 'ECOLI_ATCC25922', 'KPC2_FAV69578', 'ZYMO']:
    te = np.where(R == run)[0]
    if y[te].sum() < 20: continue
    pool = np.where((R != run) & ~np.isin(R, NOTRAIN))[0]; same = np.where((R != run) & ~np.isin(R, NOTRAIN) & np.array([KIT[r] == KIT[run] for r in R]))[0]
    for name, tri, Sc in (('pooled', pool, S2), ('samekit', same, S2), ('pooled+kit', pool, SCK)):
        if name not in os.environ.get('STRATS', 'pooled,samekit,pooled+kit').split(','): continue
        if len(same) < 1000 and name == 'samekit': continue
        m = fit(tri, Sc); ptr = predict(m, Sc, tri); pte = predict(m, Sc, te); thr = np.percentile(ptr[y[tri] == 0], 95)
        a = roc_auc_score(y[te], pte); kept = (pte[y[te] == 0] <= thr).mean(); c = (pte[y[te] == 1] > thr).mean(); t2 = np.percentile(pte[y[te] == 0], 95); c2 = (pte[y[te] == 1] > t2).mean()
        res.setdefault(name, []).append((a, kept, c, c2)); print(f'{run+" ("+KIT[run]+")":<24}{name:<12}{len(tri):8d}{a:7.3f}{100*kept:10.0f}%{100*c:11.0f}%{100*c2:33.0f}%', flush=True)
print('\nmean over held-out runs:')
for name, v in res.items(): a = np.array(v).mean(0); print(f'  {name:<12} AUC {a[0]:.3f}  good kept {100*a[1]:.0f}%  low caught {100*a[2]:.0f}%  (own-run threshold: {100*a[3]:.0f}%)   [{len(v)} runs]')
