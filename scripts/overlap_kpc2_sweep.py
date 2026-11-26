"""Precision sweep for the learned overlap seeder on KPC2 (zero-shot).  Same sample/truth as overlap_kpc2.py but fewer reads (N per flow cell) so the neighbour search is cheap; the top-10 neighbours with similarity are kept,
then pair rules are swept WITHOUT labels: K (neighbours used), similarity floor, 'crowded chunk' filter (drop a query chunk when its 10th neighbour is within d of its best = repeat-like), chain slack, V (chain length).
Truth is used only to score.  usage: overlap_kpc2_sweep.py [N]  env ENC -> results/nirai_overlap_seeder_kpc2_sweep.txt"""
import os, sys, re, pickle, itertools, collections, numpy as np, pod5, torch
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600; ENC = os.environ.get('ENC', 'overlap_enc_v2.pt'); CH = 2000
torch.set_num_threads(int(os.environ.get('THREADS', 12))); R0 = os.path.expanduser('~/rawsamble_work/'); M = R0 + 'kpc2_pool/moves/'
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_overlap_seeder_kpc2_sweep.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
order = [l.rstrip('\n').split('\t') for l in open(R0 + 'kpc2_pool/order.tsv')][1:]; fcof = {o[0]: o[3] for o in order}; arr = {o[0]: i for i, o in enumerate(order)}
lab = pickle.load(open(M + 'labels.pkl', 'rb'))
src = open(os.path.dirname(os.path.abspath(__file__)) + '/overlap_train.py').read(); ns_ = {}
exec(re.search(r'class Enc\(nn\.Module\):.*?\n(?=net = )', src, re.S).group(0), {'nn': torch.nn, 'F': torch.nn.functional, 'torch': torch}, ns_)
net = ns_['Enc'](); net.load_state_dict(torch.load(R0 + 'melb2/moves/' + ENC)); net.eval()
pick = []; cnt = collections.Counter()
for r in sorted(lab, key=lambda r: arr[r]):
    if cnt[fcof[r]] < N and lab[r]['n_samples'] >= 3000: pick.append(r); cnt[fcof[r]] += 1
P(f'\n=== KPC2 sweep, weights {ENC}, {len(pick)} sampled reads ({dict(cnt)}) ===')
sig = {}
with pod5.Reader(R0 + 'kpc2_pool/pooled.pod5') as rd:
    for rec in rd.reads(selection=pick):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float32)
E, RID, CPOS = [], [], []
with torch.no_grad():
    for k, r in enumerate(pick):
        cs = np.arange(CH // 2, len(sig[r]) - CH // 2, 500)
        if len(cs): E.append(net(torch.tensor(np.stack([sig[r][c - CH // 2: c + CH // 2] for c in cs]))[:, None])); RID += [k] * len(cs); CPOS += list(cs)
E = torch.cat(E); RID = np.array(RID); CPOS = np.array(CPOS); n = len(E); Rt = torch.tensor(RID); P(f'chunks {n}')
# per query chunk: top-10 (sim, target read, target chunk pos)
KM = 10; Q = []   # Q[a] = (i, ci, sims[10], rids[10], cposs[10])
for a in range(0, n, 512):
    S = E[a:a + 512] @ E.T; qr = Rt[a:a + 512][:, None]; S[(Rt[None, :] >= qr)] = -9; sc, ix = torch.topk(S, KM, dim=1)
    for u in range(len(ix)): Q.append((int(qr[u]), int(CPOS[a + u]), sc[u].numpy(), RID[ix[u].numpy()], CPOS[ix[u].numpy()]))
ext = {}
for k, r in enumerate(pick):
    v = lab[r]; ok = v['base_ref'] >= 0; ext[k] = (v['strand'], v['chrom'], int(v['base_ref'][ok].min()), int(v['base_ref'][ok].max()))
tp = {}
for i in range(len(pick)):
    for j in range(i):
        if ext[i][0] == ext[j][0] and ext[i][1] == ext[j][1]:
            o = min(ext[i][3], ext[j][3]) - max(ext[i][2], ext[j][2])
            if o > 0: tp[(i, j)] = o
def chain_len(v, tol, slack):
    v = sorted(set(v)); best = 0
    for a in range(len(v)):
        c = 1
        for b in range(a + 1, len(v)):
            di, dj = v[b][0] - v[a][0], v[b][1] - v[a][1]
            if di > 0 and dj > 0 and abs(dj - di) <= tol * di + slack: c += 1
        best = max(best, c)
    return best
P('rule: K neighbours, sim floor, crowd d (drop chunk if its 10th neighbour sim >= best - d), tol/slack of chain; V = chain length')
rows = []
for K, floor, crowd, (tol, slack) in itertools.product((1, 3, 5), (None, 0.85, 0.92), (None, 0.15, 0.08), ((0.1, 600), (0.1, 300), (0.05, 300))):
    votes = collections.defaultdict(list)
    for i, ci, sims, rids, cps in Q:
        if crowd is not None and sims[KM - 1] >= sims[0] - crowd: continue
        for k in range(K):
            if sims[k] > -5 and (floor is None or sims[k] >= floor): votes[(i, int(rids[k]))].append((ci, int(cps[k])))
    ch = {p: chain_len(v, tol, slack) for p, v in votes.items() if len(v) >= 3}
    for V in (3, 4, 5, 6):
        rep = [p for p, c in ch.items() if c >= V]
        if not rep: continue
        rs = set(rep); pr = np.mean([tp.get(p, 0) >= 100 for p in rep]); rc = lambda a, b: 100 * np.mean([p in rs for p, o in tp.items() if a <= o < b])
        rows.append((K, floor, crowd, tol, slack, V, len(rep), 100 * pr, rc(300, 500), rc(500, 1000), rc(1000, 2000), rc(2000, 10**9)))
for r in sorted(rows, key=lambda r: -r[7]):
    if r[9] >= 40: P(f'K={r[0]} floor={r[1]} crowd={r[2]} tol={r[3]} slack={r[4]} V={r[5]}: reported {r[6]:6d} precision {r[7]:5.1f}% | recall 300-500 {r[8]:5.1f} 500-1000 {r[9]:5.1f} 1-2kb {r[10]:5.1f} 2kb+ {r[11]:5.1f}')
P('SWEEP_DONE')
