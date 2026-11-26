"""Pair-level test of the learned overlap seeder against Rawsamble (held-out region of melb2 chromosome_1, reads never seen in training).
Causal like a live run: each read (in arrival order) queries a bank of chunks from EARLIER reads only.  A pair (new read i, earlier read j) is REPORTED when >= V query chunks of i have a top-K neighbour in j
with a consistent offset.  Truth = overlap of the reads' aligned reference extents (same strand).  Rawsamble = the reads' online.paf records (j listed as a target of i).
Reports: recall by true overlap length (ours vs Rawsamble), and precision of reported pairs (true overlap >= 100 bases).   usage: overlap_pairs.py [V] [K]   -> results/nirai_overlap_seeder_pairs_melb2.txt"""
import os, sys, pickle, collections, numpy as np, pod5, torch
sys.argv += [] ; V = int(sys.argv[1]) if len(sys.argv) > 1 else 2; K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
torch.set_num_threads(20); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); W = os.path.expanduser('~/rawsamble_work/melb2/'); CH = 2000
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_overlap_seeder_pairs_melb2.txt'), 'a'); print_tag = os.environ.get('ENC', 'overlap_enc.pt')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
import importlib.util, re
src = open(os.path.dirname(os.path.abspath(__file__)) + '/overlap_train.py').read(); ns = {}
exec(re.search(r'class Enc\(nn\.Module\):.*?\n(?=net = )', src, re.S).group(0), {'nn': torch.nn, 'F': torch.nn.functional, 'torch': torch}, ns)
net = ns['Enc'](); net.load_state_dict(torch.load(W + 'moves/' + os.environ.get('ENC', 'overlap_enc.pt'))); net.eval()
lab = pickle.load(open(W + 'moves/labels.pkl', 'rb')); clen1 = int([l.split()[1] for l in open(W + 'truth/ATCC_17802__202309/reference.fna.fai') if l.startswith('chromosome_1')][0]); LO = int(0.75 * clen1)
arrival = {l.strip(): i for i, l in enumerate(open(W + 'fc50k/stream/ids_first50000.txt'))}
test = [r for r, v in lab.items() if v['chrom'] == 'chromosome_1' and v['base_ref'][v['base_ref'] >= 0].min() >= LO and v['n_samples'] >= 3000]; test.sort(key=lambda r: arrival[r]); P(f'\n=== pair test, V={V} K={K}: {len(test)} held-out reads, causal bank ===')
sig = {}
with pod5.Reader(W + 'fc50k/stream/sorted.pod5') as rd:
    for rec in rd.reads(selection=test):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float32)
E, RID, CPOS, RK = [], [], [], []
with torch.no_grad():
    for k, r in enumerate(test):
        cs = np.arange(CH // 2, len(sig[r]) - CH // 2, 500)
        if len(cs) == 0: continue
        X = torch.tensor(np.stack([sig[r][c - CH // 2: c + CH // 2] for c in cs]))[:, None]; E.append(net(X)); RID += [k] * len(cs); CPOS += list(cs); RK += [k] * len(cs)
E = torch.cat(E); RID = np.array(RID); CPOS = np.array(CPOS); n = len(E); P(f'chunks {n}')
votes = collections.defaultdict(list)   # (i, j) -> [(c_i, c_j)]
Rt = torch.tensor(RID)
for a in range(0, n, 1024):
    S = E[a:a + 1024] @ E.T; qr = Rt[a:a + 1024][:, None]; S[(Rt[None, :] >= qr)] = -9      # earlier reads only
    sc, ix = torch.topk(S, K, dim=1)
    for qi in range(len(ix)):
        i = int(qr[qi]); ci = CPOS[a + qi]
        for rk, (s_, t) in enumerate(zip(sc[qi].tolist(), ix[qi].tolist())):
            if s_ > -5: votes[(i, int(RID[t]))].append((ci, CPOS[t], rk))
ext = {}
for k, r in enumerate(test):
    v = lab[r]; ok = v['base_ref'] >= 0; ext[k] = (v['strand'], int(v['base_ref'][ok].min()), int(v['base_ref'][ok].max()))
tp = {}
for i in range(len(test)):
    for j in range(i):
        if ext[i][0] == ext[j][0]:
            o = min(ext[i][2], ext[j][2]) - max(ext[i][1], ext[j][1])
            if o > 0: tp[(i, j)] = o
rs = {}
for l in open(W + 'fc50k/run/online.paf'):
    c = l.split('\t', 6); rs.setdefault(c[0], set())
    if c[5] != '*': rs[c[0]].add(c[5])
pos_of = {r: k for k, r in enumerate(test)}; raw = {(pos_of[q], pos_of[t]) for q, ts in rs.items() if q in pos_of for t in ts if t in pos_of}
def consistent(v, tol):
    v = sorted(set((x, y) for x, y, _ in v)); best = 0
    for a in range(len(v)):
        cnt = 1
        for b in range(a + 1, len(v)):
            di, dj = v[b][0] - v[a][0], v[b][1] - v[a][1]
            if di > 0 and dj > 0 and abs(dj - di) <= tol * di + 600: cnt += 1
        best = max(best, cnt)
    return best
P('settings: K = neighbours per chunk, V = consistent chunk matches, tol = allowed speed difference.  precision = reported pairs with true overlap >= 100 bases')
P(f'Rawsamble: reported {len(raw)}, precision {100*np.mean([tp.get(p,0)>=100 for p in raw]):.1f}%; recall 300-500 {100*np.mean([p in raw for p in tp if 300<=tp[p]<500]):.1f}%, 500-1000 {100*np.mean([p in raw for p in tp if 500<=tp[p]<1000]):.1f}%, 1-2kb {100*np.mean([p in raw for p in tp if 1000<=tp[p]<2000]):.1f}%')
for K_ in (1, 3, 5, 20):
    for V_ in (2, 3, 4, 6):
        for tol in (0.2, 0.1):
            rep = {p for p, v in votes.items() if len([x for x in v if x[2] < K_]) >= V_ and consistent([x for x in v if x[2] < K_], tol) >= V_}
            if not rep: continue
            pr = np.mean([tp.get(p, 0) >= 100 for p in rep]); rc = lambda a, b: 100 * np.mean([p in rep for p, o in tp.items() if a <= o < b])
            P(f'K={K_:<2} V={V_} tol={tol}: reported {len(rep):>7} precision {100*pr:5.1f}% | recall 0-300 {rc(0,300):5.1f}  300-500 {rc(300,500):5.1f}  500-1000 {rc(500,1000):5.1f}  1-2kb {rc(1000,2000):5.1f}  2kb+ {rc(2000,1e9):5.1f}')
P('PAIRS_DONE')
