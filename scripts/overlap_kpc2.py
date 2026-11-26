"""Zero-shot test of the learned overlap seeder on a NEW organism (K. pneumoniae KPC2, 4 flow cells incl. rapid-barcode), never seen in training (trained on melb2 only).
Labels: Dorado fast v5.2.0 --emit-moves (kpc2_pool/moves/kpc2.bam) + minimap2 to kpc2_truth reference. Sample: first N reads (pooled arrival order) of each flow cell, causal bank = earlier sampled reads.
Same pair rule as overlap_pairs.py (K=3, V=3, tol 0.1 and K=5). Rawsamble = pooled online.paf records among the sampled reads.   usage: overlap_kpc2.py [N_per_flowcell]  env ENC=weights file in melb2/moves/
-> results/nirai_overlap_seeder_kpc2.txt"""
import os, sys, re, pickle, subprocess, collections, numpy as np, pod5, pysam, torch
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; ENC = os.environ.get('ENC', 'overlap_enc_v2.pt'); CH = 2000
torch.set_num_threads(16); R0 = os.path.expanduser('~/rawsamble_work/'); M = R0 + 'kpc2_pool/moves/'; REF = R0 + 'kpc2_truth/KPC2__202310/reference.fna'
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_overlap_seeder_kpc2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
order = [l.rstrip('\n').split('\t') for l in open(R0 + 'kpc2_pool/order.tsv')][1:]; fcof = {o[0]: o[3] for o in order}; arr = {o[0]: i for i, o in enumerate(order)}
# ---- labels (cached)
lp = M + 'labels.pkl'
if not os.path.exists(lp):
    fq = M + 'kpc2.fastq'; info = {}
    with pysam.AlignmentFile(M + 'kpc2.bam', 'rb', check_sq=False) as bam, open(fq, 'w') as o:
        for r in bam.fetch(until_eof=True):
            if r.is_secondary or r.is_supplementary or r.query_name not in arr: continue
            mv = r.get_tag('mv'); stride = mv[0]; ts = r.get_tag('ts') if r.has_tag('ts') else 0; mv = np.array(mv[1:], np.int8); starts = ts + stride * np.flatnonzero(mv)
            if len(starts) != len(r.query_sequence): continue
            info[r.query_name] = (r.get_tag('ns') if r.has_tag('ns') else 0, starts.astype(np.int32)); o.write(f'@{r.query_name}\n{r.query_sequence}\n+\n{"".join(chr(q + 33) for q in r.query_qualities)}\n')
    subprocess.run(f'minimap2 -a -x map-ont -t 8 --secondary=no {REF} {fq} > {M}kpc2.sam 2> {M}mm2.err', shell=True, check=True); out = {}
    with pysam.AlignmentFile(M + 'kpc2.sam', 'r') as s:
        for r in s.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary or r.query_name not in info: continue
            ns, starts = info[r.query_name]; L = len(starts)
            if r.mapping_quality < 20 or r.query_alignment_length < 0.5 * L: continue
            ref = np.full(L, -1, np.int32)
            for q, t in r.get_aligned_pairs(matches_only=True): ref[(L - 1 - q) if r.is_reverse else q] = t
            out[r.query_name] = dict(chrom=r.reference_name, strand='-' if r.is_reverse else '+', n_samples=ns, base_sample=starts, base_ref=ref)
    pickle.dump(out, open(lp, 'wb'))
lab = pickle.load(open(lp, 'rb'))
src = open(os.path.dirname(os.path.abspath(__file__)) + '/overlap_train.py').read(); ns_ = {}
exec(re.search(r'class Enc\(nn\.Module\):.*?\n(?=net = )', src, re.S).group(0), {'nn': torch.nn, 'F': torch.nn.functional, 'torch': torch}, ns_)
net = ns_['Enc'](); net.load_state_dict(torch.load(R0 + 'melb2/moves/' + ENC)); net.eval()
pick = []; cnt = collections.Counter()
for r in sorted(lab, key=lambda r: arr[r]):
    f = fcof[r]
    if cnt[f] < N and lab[r]['n_samples'] >= 3000: pick.append(r); cnt[f] += 1
P(f'\n=== KPC2 zero-shot, weights {ENC}: labelled reads {len(lab)} of {len(order)}; sampled {len(pick)} ({dict(cnt)}), causal bank in pooled arrival order ===')
sig = {}
with pod5.Reader(R0 + 'kpc2_pool/pooled.pod5') as rd:
    for rec in rd.reads(selection=pick):
        x = rec.signal.astype(np.float32); m = np.median(x); sig[str(rec.read_id)] = np.clip((x - m) / (1.4826 * np.median(np.abs(x - m)) + 1e-6), -4, 4).astype(np.float32)
E, RID, CPOS = [], [], []
with torch.no_grad():
    for k, r in enumerate(pick):
        cs = np.arange(CH // 2, len(sig[r]) - CH // 2, 500)
        if len(cs) == 0: continue
        E.append(net(torch.tensor(np.stack([sig[r][c - CH // 2: c + CH // 2] for c in cs]))[:, None])); RID += [k] * len(cs); CPOS += list(cs)
E = torch.cat(E); RID = np.array(RID); CPOS = np.array(CPOS); n = len(E); Rt = torch.tensor(RID); P(f'chunks {n}')
KM = 5; votes = collections.defaultdict(list)
for a in range(0, n, 512):
    S = E[a:a + 512] @ E.T; qr = Rt[a:a + 512][:, None]; S[(Rt[None, :] >= qr)] = -9
    sc, ix = torch.topk(S, KM, dim=1)
    for qi in range(len(ix)):
        i = int(qr[qi]); ci = CPOS[a + qi]
        for rk, (s_, t) in enumerate(zip(sc[qi].tolist(), ix[qi].tolist())):
            if s_ > -5: votes[(i, int(RID[t]))].append((ci, CPOS[t], rk))
ext = {}
for k, r in enumerate(pick):
    v = lab[r]; ok = v['base_ref'] >= 0; ext[k] = (v['strand'], v['chrom'], int(v['base_ref'][ok].min()), int(v['base_ref'][ok].max()))
tp = {}
for i in range(len(pick)):
    for j in range(i):
        if ext[i][0] == ext[j][0] and ext[i][1] == ext[j][1]:
            o = min(ext[i][3], ext[j][3]) - max(ext[i][2], ext[j][2])
            if o > 0: tp[(i, j)] = o
pos_of = {r: k for k, r in enumerate(pick)}; raw = set()
for l in open(R0 + 'kpc2_pool/online.paf'):
    c = l.split('\t', 6)
    if c[0] in pos_of and c[5] != '*' and c[5] in pos_of: raw.add((pos_of[c[0]], pos_of[c[5]]))
def consistent(v, tol):
    v = sorted(set((x, y) for x, y, _ in v)); best = 0
    for a in range(len(v)):
        cnt_ = 1
        for b in range(a + 1, len(v)):
            di, dj = v[b][0] - v[a][0], v[b][1] - v[a][1]
            if di > 0 and dj > 0 and abs(dj - di) <= tol * di + 600: cnt_ += 1
        best = max(best, cnt_)
    return best
fcs = sorted(cnt); qfc = {i: fcof[pick[i]] for i in range(len(pick))}
def line(tag, rep):
    pr = np.mean([tp.get(p, 0) >= 100 for p in rep]) if rep else float('nan'); s = f'{tag}: reported {len(rep)} precision {100*pr:.1f}% | recall (all flow cells) '
    s += '  '.join(f'{a//1000 if a>=1000 else a}-{b//1000 if b<1e9 and b>=1000 else b}: {100*np.mean([p in rep for p, o in tp.items() if a <= o < b]):.1f}' for a, b in [(300, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)])
    P(s)
    for f in fcs:
        sub = {p for p in rep if qfc[p[0]] == f}; ps = np.mean([tp.get(p, 0) >= 100 for p in sub]) if sub else float('nan')
        P(f'    {f}: precision {100*ps:.1f}%; recall 500-1000 {100*np.mean([p in rep for p, o in tp.items() if qfc[p[0]]==f and 500<=o<1000]):.1f}, 1-2kb {100*np.mean([p in rep for p, o in tp.items() if qfc[p[0]]==f and 1000<=o<2000]):.1f}, 2kb+ {100*np.mean([p in rep for p, o in tp.items() if qfc[p[0]]==f and o>=2000]):.1f}  (true pairs {sum(1 for p in tp if qfc[p[0]]==f)})')
P('true overlap buckets in bases: 300-500, 500-1000, 1-2k, 2-4k, 4k+')
line('Rawsamble', raw)
for K_, V_ in ((3, 3), (5, 3), (3, 2)):
    rep = {p for p, v in votes.items() if len([x for x in v if x[2] < K_]) >= V_ and consistent([x for x in v if x[2] < K_], 0.1) >= V_}; line(f'seeder K={K_} V={V_} tol=0.1', rep)
P('KPC2_DONE')
