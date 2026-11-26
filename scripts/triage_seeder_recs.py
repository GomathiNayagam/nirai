"""Turn the learned overlap seeder into overlap records for the triage rule (melb2 first 50,000 reads, causal).
Cross-fitted: model q (overlap_enc_cv{q}.pt, trained with quarter q of every chromosome held out) answers only for reads whose aligned midpoint lies in quarter q (reads without a place use model index%4).
Each model embeds ALL reads (chunks of 2000 samples, stride 500); a query read searches the chunks of EARLIER reads only; pair (i,j) is reported when >= V chunk matches (top-K neighbours) are consistent (tol 0.1).
Output records per read: (target read, query start, query end) in event units (samples scaled to the read's PAF length) from the chunks of the best consistent chain +-1000 samples, like nirai.triage.read_paf.
usage: triage_seeder_recs.py -> melb2/moves/seeder_recs_K{K}V{V}.pkl for (K,V) in ((3,3),(5,3))"""
import os, re, sys, pickle, collections, numpy as np, pod5, torch
torch.set_num_threads(int(os.environ.get('THREADS', 24))); W = os.path.expanduser('~/rawsamble_work/melb2/'); CH = 2000; ARMS = ((3, 3, 0), (5, 3, 0)) if os.environ.get('CROWD') is None else ((3, 4, 0.08), (1, 3, 0.08)); KM = 10; TOL = 0.1   # (K, V, crowd): crowd d>0 drops query chunks whose 10th neighbour is within d of the best (repeat-like); env CROWD=1 builds the crowd arms
sys.path.insert(0, os.path.expanduser('~/thusumbu_repo'))
from nirai.triage import read_paf
names, qlen, _ = read_paf(W + 'fc50k/run/online.paf'); N = len(names); arr = {r: i for i, r in enumerate(names)}
lab = pickle.load(open(W + 'moves/labels.pkl', 'rb')); clen = {l.split()[0]: int(l.split()[1]) for l in open(W + 'truth/ATCC_17802__202309/reference.fna.fai')}
quarter = np.array([-1] * N)
for r, v in lab.items():
    if r in arr:
        ref = v['base_ref'][v['base_ref'] >= 0]; quarter[arr[r]] = min(3, int(((ref.min() + ref.max()) / 2) / clen[v['chrom']] * 4))
quarter = np.where(quarter < 0, np.arange(N) % 4, quarter); print('reads per quarter', np.bincount(quarter), flush=True)
src = open(os.path.dirname(os.path.abspath(__file__)) + '/overlap_train.py').read(); ns = {}
exec(re.search(r'class Enc\(nn\.Module\):.*?\n(?=net = )', src, re.S).group(0), {'nn': torch.nn, 'F': torch.nn.functional, 'torch': torch}, ns)
sig = {}; t = 0
with pod5.Reader(W + 'fc50k/stream/sorted.pod5') as rd:
    for rec in rd.reads(selection=names):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float16)
print('signals', len(sig), flush=True)
def chain(v, tol):
    v = sorted(set((x, y) for x, y, _ in v)); best = []
    for a in range(len(v)):
        m = [v[a]]
        for b in range(a + 1, len(v)):
            di, dj = v[b][0] - v[a][0], v[b][1] - v[a][1]
            if di > 0 and dj > 0 and abs(dj - di) <= tol * di + 600: m.append(v[b])
        if len(m) > len(best): best = m
    return best
recs = {a: {r: [] for r in names} for a in ARMS}; npair = {a: 0 for a in ARMS}   # key (K,V,crowd)
for q in range(4):
    net = ns['Enc'](); net.load_state_dict(torch.load(W + f'moves/overlap_enc_cv{q}.pt')); net.eval(); E, RID, CPOS = [], [], []
    with torch.no_grad():
        for k, r in enumerate(names):
            x = sig[r]; cs = np.arange(CH // 2, len(x) - CH // 2, 500)
            if len(cs) == 0: continue
            E.append(net(torch.tensor(np.stack([x[c - CH // 2: c + CH // 2].astype(np.float32) for c in cs]))[:, None])); RID += [k] * len(cs); CPOS += list(cs)
    E = torch.cat(E); RID = np.array(RID); CPOS = np.array(CPOS); Rt = torch.tensor(RID); qi_all = np.where(quarter[RID] == q)[0]; print(f'model {q}: chunks {len(E)}, query chunks {len(qi_all)}', flush=True)
    votes = collections.defaultdict(list)
    for a in range(0, len(qi_all), 512):
        ids = torch.tensor(qi_all[a:a + 512]); S = E[ids] @ E.T; S[(Rt[None, :] >= Rt[ids][:, None])] = -9; sc, ix = torch.topk(S, KM, dim=1)
        for u in range(len(ids)):
            i = int(Rt[ids[u]]); ci = CPOS[int(ids[u])]
            for rk, (s_, tt) in enumerate(zip(sc[u].tolist(), ix[u].tolist())):
                if s_ > -5: votes[(i, int(RID[tt]))].append((ci, CPOS[tt], rk, float(sc[u][0] - sc[u][KM - 1])))
    for (K, V, CR) in ARMS:
        for (i, j), v in votes.items():
            vv = [x[:3] for x in v if x[2] < K and (CR == 0 or x[3] > CR)]
            if len(vv) < V: continue
            c = chain(vv, TOL)
            if len(c) < V: continue
            r = names[i]; ns_ = len(sig[r]); lo = max(0, min(x[0] for x in c) - CH // 2); hi = min(ns_, max(x[0] for x in c) + CH // 2); f = qlen[r] / ns_
            recs[(K, V, CR)][r].append((names[j], int(lo * f), int(hi * f))); npair[(K, V, CR)] += 1
    del E, votes
for (K, V, CR) in ARMS:
    pickle.dump(recs[(K, V, CR)], open(W + f"moves/seeder_recs_K{K}V{V}{'c%g' % CR if CR else ''}.pkl", 'wb')); print(f'K={K} V={V} crowd={CR}: {npair[(K, V, CR)]} pairs; reads with a record {sum(1 for r in names if recs[(K, V, CR)][r])}', flush=True)
print('RECS_DONE')
