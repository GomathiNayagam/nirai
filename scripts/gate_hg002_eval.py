"""HG002 PromethION smoke test: does a gate trained on PromethION/ligation data work on a flow cell it has never seen?
Data: 4 HG002 runs from ONT open data giab_2023.05 (2 physical flow cells, each run twice): PAO89685 = runs 0424, 0428; PAO83395 = runs 0429, 0503.  Label = MinKNOW pass/fail FOLDER (fail = 1).
Splits are by FLOW CELL (never by run, since both runs of a flow cell share the same pores).
  A  train on PAO89685 only -> test PAO83395        B  train on PAO83395 only -> test PAO89685       (PromethION-only model)
  C/D  the same, but training also includes base runs + 5 hac species (native barcode kit), i.e. a mixed model; also tested on melb2 (held out) to check the MinION result does not get worse.
  Z  zero-shot: shipped-style model (base + hac species + melb2, no HG002) on each HG002 flow cell.
The reference for run-relative features is 500 random reads per run.  Reported: AUC, catch@98/99.8 with the test set's own good-read threshold (ranking quality; the fail share here is ~50% by design, so precision is NOT reported).
usage: gate_hg002_eval.py [--smoke | --quick]   (--quick: smoke settings, HG002-only models only, 4 threads, own results file)   (--smoke: 1 seed, 3 epochs, <=8000 reads per training run)"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.gate_cnn import Net, normalize, scalars, relative
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4 if '--quick' in sys.argv else 8); W = os.path.expanduser('~/rawsamble_work/'); V = W + 'varied/'; REPO = os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]
QUICK = '--quick' in sys.argv; SMOKE = '--smoke' in sys.argv or QUICK; EP = 3 if SMOKE else 6; SEEDS = (0,) if SMOKE else (0, 1, 2); CAP = 8000 if SMOKE else 30000; QC = 10
RUNS = {'0424': ('PAO89685', '20230424_1302_3H_PAO89685_2264ba8c'), '0428': ('PAO89685', '20230428_1310_3H_PAO89685_c9d0d53f'), '0429': ('PAO83395', '20230429_1600_2E_PAO83395_124388f5'), '0503': ('PAO83395', '20230503_1206_2E_PAO83395_ba06e1bb')}
rng = np.random.default_rng(0); DS = {}
def add(name, raw, y, kit, ref_idx, group, st=None):
    S = scalars(raw); ref = np.median(S[ref_idx], axis=0); DS[name] = dict(X=normalize(raw), S=relative(S, ref), y=y.astype(np.float32), kit=kit, group=group, st=st); print(f'{name:<16} {len(y):6d} reads, low/fail {100 * y.mean():4.1f}%, kit {kit}, group {group}', flush=True)
z = np.load(W + 'gate_loro/kit_data.npz', allow_pickle=True); e = np.load(W + 'gate_loro/failaug/extra.npz', allow_pickle=True)
for run in ('KPC2_FAX39616', 'ECOLI_ATCC25922', 'RDH275', 'KPC2_FAX76295', 'KPC2_FAX39617'):
    m = z['R'] == run; raw, Q, I = z['XR'][m], z['Q'][m].astype(float), z['I'][m]
    if run == 'RDH275': raw = np.concatenate([raw, e['XE']]); Q = np.concatenate([Q, e['QE']]); I = np.concatenate([I, 100000 + np.arange(len(e['QE']))])
    ok = ~np.isnan(Q); raw, Q, I = raw[ok], Q[ok], I[ok]; add(run, raw, (Q < QC), 0, np.argsort(I)[:500], 'base')
for n in ('CJEJ_33560', 'LIVAN_19119', 'SAUR_BPH2947', 'MTB_AMtb1', 'SALM_10708'):
    p = np.load(V + n + '/prefix5000.npz', allow_pickle=True); Q = p['Q'].astype(float); ok = ~np.isnan(Q); add(n, p['X'][ok], Q[ok] < QC, 0, np.argsort(p['I'][ok])[:500], 'hac')
m5 = np.load(W + 'melb2/gate_eval/lc/melb2_prefix5000.npz', allow_pickle=True); Q = m5['QM'].astype(float); ok = ~np.isnan(Q); add('MELB2_Vpara', m5['XM'][ok], Q[ok] < QC, 0, np.argsort(m5['IM'][ok])[:500], 'melb2')
for k, (fc, d) in RUNS.items():
    f = V + 'HG002_ALL/' + d + '/prefix5000.npz'
    if not os.path.exists(f): print('MISSING', f); continue
    p = np.load(f); add(f'HG002_{k}_{fc}', p['X'], p['FAIL'], 2, np.random.default_rng(1).choice(len(p['FAIL']), 500, replace=False), fc, st=p['ST'])
def stack(names):
    X, S, y, K = [], [], [], []
    for n in names:
        d = DS[n]; idx = np.arange(len(d['y']))
        if d['group'] != 'base' and len(idx) > CAP: idx = np.sort(rng.choice(idx, CAP, replace=False))
        X.append(d['X'][idx]); S.append(d['S'][idx]); y.append(d['y'][idx]); K.append(np.full(len(idx), d['kit']))
    return np.concatenate(X), np.concatenate(S), np.concatenate(y), np.concatenate(K)
def feats(S, K, mean, std): return np.hstack([(S - mean) / std, np.eye(3, dtype=np.float32)[K]]).astype(np.float32)
def fit(names, seed):
    X, S, y, K = stack(names); mean, std = S.mean(0), S.std(0) + 1e-9; SC = feats(S, K, mean, std); torch.manual_seed(seed); np.random.seed(seed)
    m = Net(SC.shape[1]); pw = min((1 - y.mean()) / y.mean(), 20.0); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw)); t0 = time.time()
    for ep in range(EP):
        m.train(); perm = np.random.permutation(len(y)); tot = 0
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]; loss = lf(m(torch.from_numpy(X[b].astype(np.float32))[:, None], torch.from_numpy(SC[b])), torch.from_numpy(y[b])); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        print(f'   epoch {ep + 1}/{EP} loss {tot / len(y):.4f} ({time.time() - t0:.0f}s)', flush=True)
    print(f'   trained on {len(y)} reads ({int(y.sum())} low/fail)', flush=True); return m, mean, std
def predict(m, X, SC):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(X), 1024): o.append(torch.sigmoid(m(torch.from_numpy(X[i:i+1024].astype(np.float32))[:, None], torch.from_numpy(SC[i:i+1024]))).numpy())
    return np.concatenate(o)
def tm_stats(p, y, st, nb=20):
    """Time-matched check: split the run into nb time bins (by start sample); AUC inside each bin that has >=100 pass and >=100 fail reads, averaged (weight = smaller class).  Also the AUC of a 'time only' score (the fail share of the read's own time bin, in-sample): how much of the label time alone explains."""
    edges = np.quantile(st, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(edges, st, side='right') - 1, 0, nb - 1); a, w, prior = [], [], np.zeros(len(y))
    for i in range(nb):
        m = b == i; yy = y[m]
        if m.any(): prior[m] = yy.mean()
        if (yy == 1).sum() >= 100 and (yy == 0).sum() >= 100: a.append(roc_auc_score(yy, p[m])); w.append(min((yy == 1).sum(), (yy == 0).sum()))
    return (round(float(np.average(a, weights=w)), 3) if a else None), len(a), round(float(roc_auc_score(y, prior)), 3)
def evaluate(m, mean, std, name, kit=None):
    d = DS[name]; k = d['kit'] if kit is None else kit; p = predict(m, d['X'], feats(d['S'], np.full(len(d['y']), k), mean, std)); y = d['y']; g, l = p[y == 0], p[y == 1]
    o = dict(test=name, n=int(len(y)), low_pct=round(100 * y.mean(), 1), auc=round(float(roc_auc_score(y, p)), 3), kit_fed=int(k))
    for gk in (98, 99.8): o[f'catch@{gk}'] = round(float(100 * (l > np.percentile(g, gk)).mean()), 1)
    if d['st'] is not None: o['auc_timematched'], o['n_time_bins_used'], o['auc_time_only'] = tm_stats(p, y, d['st'])
    return o
OUT = open(REPO + ('/results/nirai_gate_hg002_promethion_quick.jsonl' if QUICK else '/results/nirai_gate_hg002_promethion.jsonl'), 'a'); base = [n for n in DS if DS[n]['group'] == 'base']; hac = [n for n in DS if DS[n]['group'] == 'hac']
hg = {fc: [n for n in DS if DS[n]['group'] == fc] for fc in ('PAO89685', 'PAO83395')}
def run(tag, train, tests, seed):
    print(f'== {tag} seed {seed}: train {train}', flush=True); m, mean, std = fit(train, seed); trained = {DS[n]['kit'] for n in train}
    for t in tests:
        for k in ([None] + ([next(iter(trained))] if len(trained) == 1 and DS[t]['kit'] not in trained else [])):      # a model trained on one kit is also tested with that kit fed in
            o = evaluate(m, mean, std, t, k); o.update(setting=tag, seed=seed, smoke=SMOKE); OUT.write(json.dumps(o, default=float) + '\n'); OUT.flush(); print('  ', {kk: v for kk, v in o.items() if kk not in ('setting', 'smoke')}, flush=True)
for seed in SEEDS:
    if not QUICK: run('Z zero-shot (base+species+melb2, no HG002)', base + hac + ['MELB2_Vpara'], hg['PAO89685'] + hg['PAO83395'], seed)
    for a, b, tg in (('PAO89685', 'PAO83395', 'A'), ('PAO83395', 'PAO89685', 'B')):
        if hg[a] and hg[b]:
            run(f'{tg} HG002-only: train {a}, test {b} + MinION sets', hg[a], hg[b] + ['MELB2_Vpara'] + hac, seed)
            if not QUICK: run(f'{tg}+ mixed: train {a} + base + species, test {b} and melb2', hg[a] + base + hac, hg[b] + ['MELB2_Vpara'], seed)
