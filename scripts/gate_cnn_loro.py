"""Small 1D CNN on the first 1.0 s of raw signal (+ the noise/pA scalar features), leave-one-run-out, ladder never used for training.
low = mean Q (our hac basecalls) < QCUT. Per-read signal is median/MAD normalised (so absolute pA is not seen by the CNN); run-relative median pA enters as a scalar.
usage: gate_cnn_loro.py OUTDIR [QCUT=10]"""
import os, sys, numpy as np, pandas as pd, pod5, torch, torch.nn as nn
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.features import read_pa, signal_features
from sklearn.metrics import roc_auc_score
torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
W = os.path.expanduser('~/rawsamble_work/'); out = sys.argv[1]; QC = float(sys.argv[2]) if len(sys.argv) > 2 else 10; N = 5000
ns = {}; src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate_prefix_sweep.py')).read().split("cache = out")[0].split("W = os.path")[1]
exec("import os,numpy as np\nW = os.path" + src.replace("out = sys.argv[1]; os.makedirs(out, exist_ok=True)", "out=''"), {'os': os, 'np': np, 'pd': pd, 'pod5': pod5, 'sys': sys}, ns)
RUNS, readq, NOTRAIN = ns['RUNS'], ns['readq'], ns['NOTRAIN']
cache = out + '/cnn_data.npz'
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); X, S, Q, R, I = z['X'], z['S'], z['Q'], z['R'], z['I']
else:
    X = []; S = []; Q = []; R = []; I = []
    for run, (ps, fqs) in RUNS.items():
        q = readq(fqs); k = 0
        for p in ps:
            with pod5.Reader(p) as r:
                for rec in r.reads():
                    rid = str(rec.read_id)
                    if rid not in q: continue
                    x = read_pa(rec)
                    if len(x) < N: continue
                    x = x[:N]; med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-3
                    X.append(np.clip((x - med) / (1.4826 * mad), -6, 6).astype(np.float16)); S.append(signal_features(x) + [med]); Q.append(q[rid]); R.append(run); I.append(k); k += 1
        print(run, k, flush=True)
    X = np.stack(X); S = np.array(S, np.float32); Q = np.array(Q); R = np.array(R); I = np.array(I); np.savez(cache, X=X, S=S, Q=Q, R=R, I=I)
NP = int(os.environ.get('PREFIX_SAMPLES', N))
if NP < N:   # shorter look: the CNN sees only the first NP samples and the scalars come from the same NP samples (pa_features.tsv @0.4s, same read order)
    assert NP == 2000; X = X[:, :NP]; pf = pd.read_csv(out + '/pa_features.tsv', sep='\t'); assert len(pf) == len(X) and (pf.run.values == R).all()
    S = pf[['sd@0.4s', 'mad_diff@0.4s', 'frac_hi@0.4s', 'med_pa@0.4s']].values.astype(np.float32)
# run-relative scalars: sd, mad, frac_hi divided by the run's first-500 median; median pA minus it; plus raw log values
S2 = np.zeros_like(S)
for run in np.unique(R):
    m = R == run; first = S[m][np.argsort(I[m])[:500]]; ref = np.median(first, axis=0)
    S2[m, :3] = S[m, :3] / (ref[:3] + 1e-9); S2[m, 3] = (S[m, 3] - ref[3]) / 10
S2 = np.hstack([S2, np.log(S[:, :3] + 1e-6)]); S2 = (S2 - S2.mean(0)) / (S2.std(0) + 1e-9)
y = (Q < QC).astype(np.float32)
class CfC(nn.Module):
    """Minimal closed-form continuous-time (liquid) cell (Hasani et al. 2022): h' = sigma(-f*t)*g + (1-sigma(-f*t))*h-ish gate, time-constant learned per unit."""
    def __init__(s, i, h):
        super().__init__(); s.h = h; s.ff1 = nn.Linear(i + h, h); s.ff2 = nn.Linear(i + h, h); s.ta = nn.Linear(i + h, h); s.tb = nn.Linear(i + h, h)
    def forward(s, x, hs):
        z = torch.cat([x, hs], 1); f = torch.tanh(s.ff1(z)); g = torch.tanh(s.ff2(z)); gate = torch.sigmoid(s.ta(z) * 1.0 + s.tb(z))   # t = 1 per window
        return (1 - gate) * f + gate * g
class LNNEnc(nn.Module):
    def __init__(s, win=10, hid=64):
        super().__init__(); s.win = win; s.cell = CfC(2, hid); s.hid = hid
    def forward(s, x):                      # x [B,1,L] -> windows of `win` samples -> (mean, std) per window -> recurrent liquid cell over time
        B, _, L = x.shape; w = x[:, 0, : L // s.win * s.win].reshape(B, -1, s.win); u = torch.stack([w.mean(2), w.std(2)], 2); hs = torch.zeros(B, s.hid)
        for t in range(u.shape[1]): hs = s.cell(u[:, t], hs)
        return hs
class Net(nn.Module):
    def __init__(s, ns, sig=True):
        super().__init__(); s.sig = sig
        c = lambda i, o, k, st: nn.Sequential(nn.Conv1d(i, o, k, st, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        s.conv = LNNEnc() if os.environ.get('MODEL') == 'lnn' else nn.Sequential(c(1, 16, 9, 2), c(16, 32, 9, 4), c(32, 48, 9, 4), c(48, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.head = nn.Sequential(nn.Linear((64 if sig else 0) + ns, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, sc): return s.head(torch.cat([s.conv(x), sc], 1) if s.sig else sc).squeeze(1)
def predict(m, Xs, Ss):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(Xs), 1024): o.append(torch.sigmoid(m(torch.from_numpy(Xs[i:i+1024].astype(np.float32))[:, None], torch.from_numpy(Ss[i:i+1024]))).numpy())
    return np.concatenate(o)
def fit(tri, epochs=8, use_sig=True):
    m = Net(S2.shape[1], sig=use_sig); pw = (1 - y[tri].mean()) / y[tri].mean(); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(pw, 20.0)))
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; xb = torch.from_numpy(X[b].astype(np.float32))[:, None]
            loss = lf(m(xb, torch.from_numpy(S2[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
    return m
print(f'low = mean Q < {QC}; reads {len(y)}; low share per run: ' + ', '.join(f'{r} {100*y[R==r].mean():.1f}%' for r in np.unique(R)), flush=True)
print(f'{"held-out run":<16}{"n":>6}{"low n":>7}   CNN+scalars: AUC  good kept  low caught   |  scalars only (no signal): AUC  good kept  low caught', flush=True)
res = []
for run in np.unique(R):
    if run in NOTRAIN or y[R == run].sum() < 20: continue
    te = np.where(R == run)[0]; tr = np.where((R != run) & ~np.isin(R, NOTRAIN + [x for x in os.environ.get('EXCLUDE_TRAIN', '').split(',') if x]))[0]; row = []
    for use in ((False,) if os.environ.get('ONLY_SCALARS') else ((True,) if os.environ.get('ONLY_SIGNAL') else (True, False))):
        m = fit(tr, use_sig=use); ptr = predict(m, X[tr], S2[tr]); pte = predict(m, X[te], S2[te])
        thr = np.percentile(ptr[y[tr] == 0], 95); row += [roc_auc_score(y[te], pte), (pte[y[te] == 0] <= thr).mean(), (pte[y[te] == 1] > thr).mean()]
    if os.environ.get('ONLY_SCALARS'): row = [float('nan')] * 3 + row
    if os.environ.get('ONLY_SIGNAL'): row = row + [float('nan')] * 3
    res.append(row); print(f'{run:<16}{len(te):6d}{int(y[te].sum()):7d}   {row[0]:21.3f} {100*row[1]:8.0f}% {100*row[2]:9.0f}%   |   {row[3]:21.3f} {100*row[4]:8.0f}% {100*row[5]:9.0f}%', flush=True)
a = np.array(res).mean(0); print(f'{"mean":<29}   {a[0]:21.3f} {100*a[1]:8.0f}% {100*a[2]:9.0f}%   |   {a[3]:21.3f} {100*a[4]:8.0f}% {100*a[5]:9.0f}%')
