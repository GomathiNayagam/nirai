"""Does the 1.0 s CNN gate hold up on varied data?  Runs: 5 base runs (KPC2 x3, E. coli, RDH275 + extra fail reads), melb2 (V. parahaemolyticus), five new species from the same NanoVarBench
sets (C. jejuni, L. ivanovii, S. aureus, M. tuberculosis, Salmonella; labels = public hac calls) and HG002 on a PromethION (FLO-PRO114M, ligation kit; labels = MinKNOW mean_qscore_template).
  zero-shot : train on the 5 base runs only, test on every other dataset (2 seeds)
  leave-one-out : train on the base runs + all other hac-labelled datasets (not HG002), test on the one left out (1 seed); HG002 is left out in every fold of this kind (tested once, 'all-but-HG002')
  final     : train on the base runs + all hac-labelled datasets (NOT HG002: it does not transfer and its folder labels are weak), save models/nirai_gate_cnn_1.0s.pt, then check the saved model reproduces the in-script scores through nirai.gate_cnn.
Reads with no label are dropped everywhere.  Low = mean Q < 10.   usage: gate_varied_eval.py [--final-only]   (--final-only skips the zero-shot / leave-one-out runs)"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.gate_cnn import Net, normalize, scalars, relative, save_cnn_gate, load_cnn_gate
from sklearn.metrics import roc_auc_score
torch.set_num_threads(8); W = os.path.expanduser('~/rawsamble_work/'); V = W + 'varied/'; REPO = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]; CAP = 30000; QC = 10
rng = np.random.default_rng(0); DS = {}
def add(name, raw, Q, I, kit, group):
    ok = ~np.isnan(Q); raw, Q, I = raw[ok], Q[ok], I[ok]; S = scalars(raw); ref = np.median(S[np.argsort(I)[:500]] if group != 'hg002' else S[np.random.default_rng(1).choice(len(S), min(500, len(S)), replace=False)], axis=0)   # HG002: the 6 files are not a contiguous stretch of the run, so the reference uses 500 random reads
    DS[name] = dict(X=normalize(raw), S=relative(S, ref), y=(Q < QC).astype(np.float32), kit=kit, group=group, ref=ref); print(f'{name:<18} {len(Q):6d} labelled reads, low {100 * (Q < QC).mean():4.1f}%, kit {kit}, group {group}', flush=True)
z = np.load(W + 'gate_loro/kit_data.npz', allow_pickle=True); e = np.load(W + 'gate_loro/failaug/extra.npz', allow_pickle=True)
for run in ('KPC2_FAX39616', 'ECOLI_ATCC25922', 'RDH275', 'KPC2_FAX76295', 'KPC2_FAX39617'):
    m = z['R'] == run; raw, Q, I = z['XR'][m], z['Q'][m], z['I'][m]
    if run == 'RDH275': raw = np.concatenate([raw, e['XE']]); Q = np.concatenate([Q, e['QE']]); I = np.concatenate([I, 100000 + np.arange(len(e['QE']))])
    add(run, raw, Q.astype(float), I, 0, 'base')
m5 = np.load(W + 'melb2/gate_eval/lc/melb2_prefix5000.npz', allow_pickle=True); add('MELB2_Vpara', m5['XM'], m5['QM'].astype(float), m5['IM'], 0, 'hac')
for n, d, kit in (('CJEJ_33560', 'CJEJ_33560', 0), ('LIVAN_19119', 'LIVAN_19119', 0), ('SAUR_BPH2947', 'SAUR_BPH2947', 0), ('MTB_AMtb1', 'MTB_AMtb1', 0), ('SALM_10708', 'SALM_10708', 0), ('HG002_PromethION', 'HG002_PRO', 2)):
    f = V + d + '/prefix5000.npz'
    if os.path.exists(f): p = np.load(f, allow_pickle=True); add(n, p['X'], p['Q'].astype(float), p['I'], kit, 'hg002' if 'HG002' in n else 'hac')
    else: print('MISSING', f)
def stack(names, cap=True):
    X, S, y, K = [], [], [], []
    for n in names:
        d = DS[n]; idx = np.arange(len(d['y']))
        if cap and d['group'] != 'base' and len(idx) > CAP: idx = np.sort(rng.choice(idx, CAP, replace=False))
        X.append(d['X'][idx]); S.append(d['S'][idx]); y.append(d['y'][idx]); K.append(np.full(len(idx), d['kit']))
    return np.concatenate(X), np.concatenate(S), np.concatenate(y), np.concatenate(K)
def feats(S, K, mean, std): return np.hstack([(S - mean) / std, np.eye(3, dtype=np.float32)[K]]).astype(np.float32)
def fit(names, seed, epochs=6):
    X, S, y, K = stack(names); mean, std = S.mean(0), S.std(0) + 1e-9; SC = feats(S, K, mean, std); torch.manual_seed(seed); np.random.seed(seed)
    m = Net(SC.shape[1]); pw = min((1 - y.mean()) / y.mean(), 20.0); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw)); t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(y))
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; loss = lf(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step()
    print(f'   trained on {len(y)} reads ({int(y.sum())} low) in {time.time() - t0:.0f}s', flush=True); return m, mean, std, X, SC, y
def predict(m, X, SC):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(X), 1024): o.append(torch.sigmoid(m(torch.from_numpy(X[i:i+1024].astype(np.float32))[:, None], torch.from_numpy(SC[i:i+1024]))).numpy())
    return np.concatenate(o)
def evaluate(m, mean, std, name, p_train_good):
    d = DS[name]; p = predict(m, d['X'], feats(d['S'], np.full(len(d['y']), d['kit']), mean, std)); y = d['y']
    if y.sum() == 0 or y.sum() == len(y): return None
    g, l = p[y == 0], p[y == 1]; o = dict(test=name, n=int(len(y)), low_pct=round(100 * y.mean(), 1), auc=round(float(roc_auc_score(y, p)), 3))
    for gk in (98, 99.8): o[f'catch@{gk}'] = round(float(100 * (l > np.percentile(g, gk)).mean()), 1)               # own-run threshold: ranking quality only
    for gk in (98, 99.8):                                                                                           # threshold carried from the training good reads: realistic
        thr = np.percentile(p_train_good, gk); fl = p > thr; o[f'carried{gk}_goodflag%'] = round(float(100 * (g > thr).mean()), 2); o[f'carried{gk}_badcaught%'] = round(float(100 * (l > thr).mean()), 1); o[f'carried{gk}_precision%'] = round(float(100 * y[fl].mean()), 1) if fl.sum() else None; o[f'carried{gk}_precision%'] = None if o[f'carried{gk}_precision%'] is None else float(o[f'carried{gk}_precision%'])
    return o
OUT = open(REPO + '/results/nirai_gate_varied_eval.jsonl', 'a'); base = [n for n in DS if DS[n]['group'] == 'base']; hac = [n for n in DS if DS[n]['group'] == 'hac']; hg = [n for n in DS if DS[n]['group'] == 'hg002']
def run(tag, train, tests, seed):
    print(f'== {tag} seed {seed}: train {train}', flush=True); m, mean, std, X, SC, y = fit(train, seed); ptg = predict(m, X, SC)[y == 0]
    for t in tests:
        o = evaluate(m, mean, std, t, ptg)
        if o: o.update(setting=tag, seed=seed, n_train_sets=len(train)); OUT.write(json.dumps(o, default=float) + '\n'); OUT.flush(); print('  ', {k: v for k, v in o.items() if k not in ('setting', 'n_train_sets')}, flush=True)
    return m, mean, std
FINAL_ONLY = '--final-only' in sys.argv
for seed in (0, 1):
    if not FINAL_ONLY: run('zero-shot (5 base runs)', base, hac + hg, seed)
for t in ([] if FINAL_ONLY else hac): run(f'leave-one-out (train: base + other hac-labelled sets; test {t})', base + [h for h in hac if h != t], [t], 0)
if not FINAL_ONLY: run('all-but-HG002 (test HG002)', base + hac, hg, 0)
m, mean, std = run('final (base + hac-labelled sets, no HG002; no held-out test)', base + hac, [], 0)
os.makedirs(REPO + '/models', exist_ok=True); path = REPO + '/models/nirai_gate_cnn_1.0s.pt'
save_cnn_gate(m, 5000, mean, std, path, notes='1.0 s CNN gate, trained on 5 base runs + melb2 + 5 species (native barcode kit only; HG002 PromethION NOT included); see models/README.md'); g = load_cnn_gate(path)
d = DS['MELB2_Vpara']; ok = ~np.isnan(m5['QM'].astype(float)); raw = m5['XM'][ok][:3000].astype(np.float32); p_saved = g.score(raw, g.run_reference(raw), 'native')
p_in = 1 - predict(m, d['X'][:3000], feats(d['S'][:3000], np.zeros(3000, int), mean, std))
print('saved-model check (nirai.gate_cnn.load_cnn_gate + score vs in-script, 3000 melb2 reads): max |difference in P(pass)| = %.5f' % float(np.abs(p_saved - p_in).max()), flush=True)
