"""First learned overlap seeder (melb2, first 50,000 reads).  A small 1D CNN maps a 2000-sample signal chunk (~160 bases) to a 64-d unit vector; chunks of DIFFERENT reads at the SAME reference position
(same strand) are pulled together, other positions pushed apart (InfoNCE; extra hard negatives = the same read 300-800 bases away).  Labels = overlap_prep.py (Dorado moves + minimap2).
Held-out test = the last 25% of chromosome_1 (never seen in training, +-3 kb margin), so a model that only memorised the genome fails.
Test: chunk retrieval.  Query chunks from test reads; bank = all chunks (stride 500 samples) of OTHER test reads; hit = a retrieved chunk lies within 100 bases of the query's reference position on the same strand.
usage: overlap_train.py [steps]   -> results/nirai_overlap_seeder_melb2.txt, melb2/moves/overlap_enc.pt"""
import os, sys, time, pickle, collections, numpy as np, pod5, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('THREADS', 16))); rng = np.random.default_rng(0); torch.manual_seed(0)
W = os.path.expanduser('~/rawsamble_work/melb2/'); STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; CH = 2000; B = 96; AUG = os.environ.get('AUG', '0') == '1'; ENC = os.environ.get('ENC', 'overlap_enc.pt')
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_overlap_seeder_melb2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
lab = pickle.load(open(W + 'moves/labels.pkl', 'rb')); clen = {l.split()[0]: int(l.split()[1]) for l in open(W + 'truth/ATCC_17802__202309/reference.fna.fai')}
TEST_LO = int(0.75 * clen['chromosome_1']); HOLD = int(os.environ['HOLD']) if 'HOLD' in os.environ else None   # HOLD=q: hold out quarter q (0-3) of EVERY chromosome (cross-fitting for triage_seeder_*.py)
def in_test(ch, p):
    if HOLD is None: return ch == 'chromosome_1' and TEST_LO - 3000 <= p
    return HOLD * clen[ch] / 4 - 3000 <= p < (HOLD + 1) * clen[ch] / 4 + 3000
t0 = time.time(); sig = {}
with pod5.Reader(W + 'fc50k/stream/sorted.pod5') as rd:
    for rec in rd.reads(selection=list(lab)):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float16)
P(f'\n=== overlap seeder AUG={int(AUG)} ENC={ENC}, {len(sig)} reads loaded in {time.time()-t0:.0f}s; train steps {STEPS}, batch {B} loci; test region {'chromosome_1 >= %d' % TEST_LO if HOLD is None else 'quarter %d of each chromosome' % HOLD} ===')
R = {}
for r, v in lab.items():
    ok = v['base_ref'] >= 0; ref = v['base_ref'][ok]; s = v['base_sample'][ok]; o = np.argsort(ref, kind='stable'); ref, s = ref[o], s[o]; u = np.r_[True, np.diff(ref) > 0]; R[r] = (ref[u], s[u].astype(np.float32), v['chrom'], v['strand'])
index = collections.defaultdict(list)
for r, (ref, s, ch, st) in R.items():
    for b in range(ref[0] // 500, ref[-1] // 500 + 1): index[(ch, st, b)].append(r)
def center(r, p):
    ref, s, *_ = R[r]; c = int(np.interp(p, ref, s)); return c if CH // 2 <= c <= len(sig[r]) - CH // 2 and ref[0] - 100 <= p <= ref[-1] + 100 else None
def chunk(r, c, jit=0):
    x = sig[r][c - CH // 2 + jit: c + CH // 2 + jit].astype(np.float32); return x if len(x) == CH else None
def chunk_aug(r, c):
    # random speed change (window of CH*f samples resampled to CH), extra noise, gain/offset jitter
    f = float(np.exp(rng.uniform(np.log(0.8), np.log(1.3)))); L = int(CH * f); a = c - L // 2 + int(rng.integers(-20, 21))
    if a < 0 or a + L > len(sig[r]): return None
    x = np.interp(np.linspace(0, L - 1, CH), np.arange(L), sig[r][a:a + L].astype(np.float32))
    if rng.random() < 0.8: x = x + rng.normal(0, rng.uniform(0, 0.5), CH)
    return (x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.1)).astype(np.float32)
names = list(R)
def sample_batch():
    X, Y, Z = [], [], []
    while len(X) < B:
        a = names[rng.integers(len(names))]; ref, s, ch, st = R[a]; p = int(rng.integers(ref[0], ref[-1]))
        if in_test(ch, p) or in_test(ch, p + 800) or in_test(ch, p - 800): continue
        cands = [r for r in index[(ch, st, p // 500)] if r != a]
        if not cands: continue
        b = cands[rng.integers(len(cands))]; ca, cb = center(a, p), center(b, p); q = p + int(rng.choice([-1, 1]) * rng.integers(300, 800)); cn = center(b, q)
        if ca is None or cb is None or cn is None: continue
        x, y, z = (chunk_aug(a, ca), chunk_aug(b, cb), chunk_aug(b, cn)) if AUG else (chunk(a, ca, int(rng.integers(-20, 21))), chunk(b, cb, int(rng.integers(-20, 21))), chunk(b, cn))
        if x is None or y is None or z is None: continue
        X.append(x); Y.append(y); Z.append(z)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(np.stack(Y))[:, None], torch.tensor(np.stack(Z))[:, None]
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); L = []; c = 1
        for o, k in ((32, 9), (64, 9), (96, 9), (128, 9), (128, 5)): L += [nn.Conv1d(c, o, k, 2, k // 2), nn.BatchNorm1d(o), nn.GELU()]; c = o
        s.f = nn.Sequential(*L, nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(1024, 64))
    def forward(s, x): return F.normalize(s.f(x), dim=1)
net = Enc(); P('encoder parameters', sum(p.numel() for p in net.parameters())); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
net.train(); t0 = time.time()
for step in range(STEPS):
    x, y, z = sample_batch(); e = net(torch.cat([x, y, z])); a, b, n = e[:B], e[B:2 * B], e[2 * B:]
    la = a @ torch.cat([b, n]).T / 0.07; lb = b @ torch.cat([a, n]).T / 0.07; tgt = torch.arange(B)
    loss = F.cross_entropy(la, tgt) + F.cross_entropy(lb, tgt); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if step % 200 == 0 or step == STEPS - 1: P(f'step {step} loss {loss.item():.3f} acc {(la.argmax(1) == tgt).float().mean().item():.2f} {time.time()-t0:.0f}s')
torch.save(net.state_dict(), W + 'moves/' + ENC); net.eval()
# ---- held-out retrieval
test = [r for r in names if len(sig[r]) >= 3000 and ((R[r][2] == 'chromosome_1' and R[r][0][0] >= TEST_LO) if HOLD is None else (R[r][0][0] >= HOLD * clen[R[r][2]] / 4 and R[r][0][-1] < (HOLD + 1) * clen[R[r][2]] / 4))]; P(f'test reads fully inside the held-out region: {len(test)}')
bank, meta = [], []
with torch.no_grad():
    for st in '+-':
        pass
    for r in test:
        ref, s, ch, st = R[r]; cs = np.arange(CH // 2, len(sig[r]) - CH // 2, 500)
        if len(cs) == 0: continue
        X = torch.tensor(np.stack([chunk(r, int(c)) for c in cs]))[:, None]; bank.append(net(X).numpy())
        meta += [(r, st, float(np.interp(c, s, ref)) if s[0] <= c <= s[-1] else np.nan) for c in cs]
E = np.concatenate(bank); rd_ = np.array([m[0] for m in meta]); stn = np.array([m[1] for m in meta]); pos = np.array([m[2] for m in meta]); ok = ~np.isnan(pos); P(f'bank chunks {len(E)} (with a position {int(ok.sum())})')
q = rng.choice(np.where(ok)[0], 3000, replace=False); Et = torch.tensor(E); hits = {1: 0, 10: 0, 50: 0}; nn_ = 0; rand = []
for i in q:
    sim = Et @ Et[i]; sim[torch.tensor(rd_ == rd_[i])] = -9; sim[torch.tensor(~ok)] = -9; top = torch.topk(sim, 50).indices.numpy()
    same = (stn[top] == stn[i]) & (np.abs(pos[top] - pos[i]) <= 100); has = ((stn == stn[i]) & (rd_ != rd_[i]) & ok & (np.abs(pos - pos[i]) <= 100)).any()
    if not has: continue
    nn_ += 1
    for k in hits: hits[k] += same[:k].any()
    rand.append(((stn == stn[i]) & (rd_ != rd_[i]) & ok & (np.abs(pos - pos[i]) <= 100)).sum() / max(1, (rd_ != rd_[i]).sum()))
P(f'held-out retrieval: queries with a true partner in the bank {nn_}; recall@1 {100*hits[1]/nn_:.1f}%  @10 {100*hits[10]/nn_:.1f}%  @50 {100*hits[50]/nn_:.1f}%  (chance for a random pick: {100*np.mean(rand):.3f}% per draw)')
P('SEEDER_DONE')
