"""What does the learned overlap seeder see, and where does it fail?  (held-out region of melb2 chromosome_1, same chunks as overlap_pairs.py)
(1) top-1 retrieval recall by chunk property: read quality, local speed (samples/base), reference GC, longest homopolymer, repeat content, signal spread, position inside the read.
(2) robustness: time-stretch (speed change) and added noise on the query chunk.   (3) what the wrong top-1 neighbours are: near-miss (same strand, 100-1000 bases off), far, opposite strand, repeat.
usage: overlap_probe.py   -> results/nirai_overlap_seeder_probe_melb2.txt"""
import os, sys, re, pickle, zlib, numpy as np, pod5, pysam, torch
torch.set_num_threads(20); rng = np.random.default_rng(1); W = os.path.expanduser('~/rawsamble_work/melb2/'); CH = 2000; REFD = W + 'truth/ATCC_17802__202309/'
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_overlap_seeder_probe_melb2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
src = open(os.path.dirname(os.path.abspath(__file__)) + '/overlap_train.py').read(); ns = {}
exec(re.search(r'class Enc\(nn\.Module\):.*?\n(?=net = )', src, re.S).group(0), {'nn': torch.nn, 'F': torch.nn.functional, 'torch': torch}, ns)
net = ns['Enc'](); net.load_state_dict(torch.load(W + 'moves/' + os.environ.get('ENC', 'overlap_enc.pt'))); net.eval()
lab = pickle.load(open(W + 'moves/labels.pkl', 'rb')); fa = pysam.FastaFile(REFD + 'reference.fna'); clen1 = fa.get_reference_length('chromosome_1'); LO = int(0.75 * clen1)
rep = [(int(l.split()[1]), int(l.split()[2])) for l in open(REFD + 'ATCC_17802__202309.repetitive_regions.bed') if l.startswith('chromosome_1')]
q = {}
with open(W + 'moves/s50k.fastq') as f:
    while True:
        h = f.readline()
        if not h: break
        f.readline(); f.readline(); ql = np.frombuffer(f.readline().strip().encode(), np.uint8).astype(float) - 33; q[h[1:].split()[0]] = -10 * np.log10(np.mean(10 ** (-ql / 10)))
test = [r for r, v in lab.items() if v['chrom'] == 'chromosome_1' and v['base_ref'][v['base_ref'] >= 0].min() >= LO and v['n_samples'] >= 3000]
sig = {}
with pod5.Reader(W + 'fc50k/stream/sorted.pod5') as rd:
    for rec in rd.reads(selection=test):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float32)
Rm = {}
for r in test:
    v = lab[r]; ok = v['base_ref'] >= 0; ref = v['base_ref'][ok]; s = v['base_sample'][ok]; o = np.argsort(ref, kind='stable'); ref, s = ref[o], s[o]; u = np.r_[True, np.diff(ref) > 0]; Rm[r] = (ref[u], s[u].astype(np.float32))
bank, meta = [], []
with torch.no_grad():
    for k, r in enumerate(test):
        cs = np.arange(CH // 2, len(sig[r]) - CH // 2, 500)
        if len(cs) == 0: continue
        bank.append(net(torch.tensor(np.stack([sig[r][c - CH // 2: c + CH // 2] for c in cs]))[:, None]).numpy())
        ref, s = Rm[r]; v = lab[r]
        for c in cs:
            ok = s[0] <= c <= s[-1]; p = float(np.interp(c, s, ref)) if ok else np.nan
            spd = float((np.interp(p + 50, ref, s) - np.interp(p - 50, ref, s)) / 100) if ok and ref[0] + 50 < p < ref[-1] - 50 else np.nan
            meta.append((k, v['strand'], p, abs(spd), q[r], c / len(sig[r]), float(sig[r][c - CH // 2: c + CH // 2].std())))
E = torch.tensor(np.concatenate(bank)); M = np.array([(m[0], m[2], m[3], m[4], m[5], m[6]) for m in meta]); st = np.array([m[1] for m in meta]); rid, pos, spd, qq, frac, sd = M.T; ok = ~np.isnan(pos)
def window_stats(p):
    w = fa.fetch('chromosome_1', int(p) - 80, int(p) + 80).upper(); gc = (w.count('G') + w.count('C')) / len(w); hp = max(len(m.group(0)) for m in re.finditer(r'(.)\1*', w)); cr = len(zlib.compress(w.encode())) / len(w); inrep = any(a <= p <= b for a, b in rep)
    return gc, hp, cr, inrep
idx = rng.choice(np.where(ok & ~np.isnan(spd))[0], 12000, replace=False); res = []
for i in idx:
    sim = E @ E[i]; sim[torch.tensor(rid == rid[i])] = -9; sim[torch.tensor(~ok)] = -9; j = int(sim.argmax())
    has = ((st == st[i]) & (rid != rid[i]) & ok & (np.abs(pos - pos[i]) <= 100)).any()
    if not has: continue
    d = pos[j] - pos[i]; same = st[j] == st[i]; hit = same and abs(d) <= 100; kind = 'hit' if hit else ('opposite strand' if not same else ('near miss (100-1000 bases)' if abs(d) <= 1000 else 'far'))
    res.append((i, hit, kind, *window_stats(pos[i])))
n = len(res); hit = np.array([r[1] for r in res]); kinds = [r[2] for r in res]; ii = np.array([r[0] for r in res]); gc, hp, cr, inrep = [np.array([r[k] for r in res]) for k in (3, 4, 5, 6)]
P(f'\n=== seeder probe: {len(test)} held-out reads, {len(E)} chunks, {n} queries with a true partner; top-1 recall {100*hit.mean():.1f}% ===')
P('wrong top-1 neighbours: ' + ', '.join(f'{k} {100*np.mean([x == k for x in kinds]):.1f}%' for k in ('near miss (100-1000 bases)', 'far', 'opposite strand')))
def by(name, x, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 40: out.append(f'{a:g}-{b:g}: {100*hit[m].mean():.1f}% (n {int(m.sum())})')
    P(f'recall by {name}: ' + ' | '.join(out))
by('read mean Q (Dorado fast)', qq[ii], [0, 10, 14, 17, 20, 99]); by('local speed, samples per base', spd[ii], [0, 9, 11, 13, 16, 22, 99]); by('reference GC share', gc, [0, .3, .4, .5, .6, 1.01])
by('longest homopolymer in 160 bases', hp, [0, 4, 5, 6, 8, 99]); by('compressibility of the 160-base reference window (low = repetitive)', cr, [0, .35, .4, .45, .5, 9]); by('signal spread in chunk (sd, normalised)', sd[ii], [0, .8, 1.0, 1.2, 1.5, 9]); by('position inside read (0 = start, 1 = end)', frac[ii], [0, .1, .3, .7, .9, 1.01])
P(f'recall inside annotated repeat regions {100*hit[inrep].mean() if inrep.sum() > 20 else float("nan"):.1f}% (n {int(inrep.sum())}) vs elsewhere {100*hit[~inrep].mean():.1f}%')
# ---- robustness: stretch and noise of the query chunk
sub = ii[:1500]; P('robustness (query chunk changed, bank unchanged; recall@1 on the same 1500 queries):')
def query_emb(i, f=1.0, noise=0.0):
    r = test[int(rid[i])]; c = int(round(Rm[r][1][np.argmin(abs(Rm[r][0] - pos[i]))])); L = int(CH * f); a = c - L // 2
    if a < 0 or a + L > len(sig[r]): return None
    x = np.interp(np.linspace(0, L - 1, CH), np.arange(L), sig[r][a:a + L]) + (rng.normal(0, noise, CH) if noise else 0)
    with torch.no_grad(): return net(torch.tensor(x.astype(np.float32))[None, None])[0]
for f, nz in [(1, 0), (.8, 0), (.9, 0), (1.1, 0), (1.25, 0), (1, .3), (1, .6), (1, 1.0)]:
    h = n_ = 0
    for i in sub:
        e = query_emb(i, f, nz)
        if e is None: continue
        sim = E @ e; sim[torch.tensor(rid == rid[i])] = -9; sim[torch.tensor(~ok)] = -9; j = int(sim.argmax()); n_ += 1; h += st[j] == st[i] and abs(pos[j] - pos[i]) <= 100
    P(f'  stretch x{f} noise sd {nz}: recall@1 {100*h/n_:.1f}% (n {n_})')
P('PROBE_DONE')
