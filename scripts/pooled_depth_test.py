"""Pooled-depth stopping-signal test (4 KPC2 flow cells pooled, ~48x basecalled; same isolate).
 Part 1  variant accuracy (bcftools, SNP/indel) vs depth, random subsets (finds the plateau; where does 30x sit?)
 Part 2  overlap-estimated per-window depth (Rawsamble, same-strand, earlier reads only) vs TRUE same-strand depth (reference positions from minimap2 alignment of the basecalls)
 Part 3  stopping rule = causal triage with target D (estimated same-strand depth): achieved true depth, genome fraction covered, variant accuracy vs random subset of the same bases.
 usage: pooled_depth_test.py   (needs kpc2_pool/{pooled.fastq, online.paf, order.tsv})   Env PARTS=1,2,3"""
import sys, os, subprocess, numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__)); import var_eval_lib as V
P = '/home/sgomathi/rawsamble_work/kpc2_pool/'; REF = '/home/sgomathi/rawsamble_work/kpc2_truth/KPC2__202310/reference.fna'
PARTS = os.environ.get('PARTS', '1,2,3').split(',')
order_df = pd.read_csv(P + 'order.tsv', sep='\t')
fq = {}
with open(P + 'pooled.fastq') as f:
    rec = []
    for l in f:
        rec.append(l)
        if len(rec) == 4: fq[rec[0][1:].split()[0]] = ''.join(rec); rec = []
ids_all = [r for r in order_df.read_id if r in fq]
real = {r: len(fq[r].split('\n')[1]) for r in ids_all}
print(f'{len(ids_all)} pooled reads basecalled, {sum(real.values())/1e6:.0f} Mb', flush=True)
# genome
tl = {}
for l in open(REF + '.fai'): c = l.split('\t'); tl[c[0]] = int(c[1])
G = sum(tl.values()); off = {}; o = 0
for t, L in tl.items(): off[t] = o; o += L
# truth spans from basecall alignments
if not os.path.exists(P + 'truth_bc.paf'):
    subprocess.run(f'minimap2 -x map-ont -t 16 --secondary=no -o {P}truth_bc.paf {REF} {P}pooled.fastq', shell=True, check=True, stderr=subprocess.DEVNULL)
truth = {}
for l in open(P + 'truth_bc.paf'):
    c = l.split('\t')
    if c[0] in truth: continue
    if int(c[11]) >= 20 and int(c[10]) >= 0.5 * int(c[1]):
        truth[c[0]] = (c[4], off[c[5]] + int(c[7]), off[c[5]] + int(c[8]))
print(f'reads with reference position: {len(truth)} ({len(truth)/len(ids_all)*100:.0f}%)', flush=True)
rng = np.random.default_rng(1)
def write_fq(sel, path):
    with open(path, 'w') as o: o.writelines(fq[r] for r in sel)
def at_bases(perm_ids, target):
    c = np.cumsum([real[r] for r in perm_ids]); return perm_ids[: int(np.searchsorted(c, target)) + 1]
def score(sel, tag):
    d = P + 'ds/' + tag; os.makedirs(d, exist_ok=True); write_fq(sel, d + '/reads.fastq'); r = V.score(d + '/reads.fastq', d, threads=16)
    r['Mb'] = sum(real[x] for x in sel) / 1e6; return r
def line(label, r): print(f"{label:<34} {r['Mb']:6.0f} Mb depth {r['mean_depth']:5.1f}x | SNP P {r['SNP']['P']:6.2f} R {r['SNP']['R']:6.2f} F1 {r['SNP']['F1']:6.2f} | indel P {r['indel']['P']:6.2f} R {r['indel']['R']:6.2f} F1 {r['indel']['F1']:6.2f}", flush=True)
def mean_r(rs):
    f = lambda k, m: float(np.mean([x[k][m] for x in rs])); return dict(Mb=np.mean([x['Mb'] for x in rs]), mean_depth=np.mean([x['mean_depth'] for x in rs]), SNP={m: f('SNP', m) for m in 'P R F1'.split()}, indel={m: f('indel', m) for m in 'P R F1'.split()})

if '1' in PARTS:
    print('\n===== PART 1: variant accuracy vs depth (random subsets of pooled reads, 3 draws each) =====', flush=True)
    for dep in (2, 4, 6, 8, 10, 15, 20, 30, 40, 1000):
        rs = [score(at_bases(list(rng.permutation(ids_all)), dep * G), f'depth{dep}_{k}') for k in range(1 if dep == 1000 else 3)]
        line(f'random, target {dep}x' if dep < 1000 else 'ALL pooled reads', mean_r(rs))

# ---- machinery shared by parts 2 and 3: causal overlap records (triage_sim) ----
if '2' in PARTS or '3' in PARTS:
    os.environ['CONTIG'] = 'x'; os.environ['TRI_W'] = '1000'
    fp = P + 'ds/online_sub.paf'; S = set(ids_all); kept = {}; q0 = {}
    for l in open(P + 'online.paf'):
        c = l.split('\t')
        if c[0] not in S: continue
        kept.setdefault(c[0], []); q0.setdefault(c[0], c[1])
        if c[5] != '*' and c[5] in S: kept[c[0]].append(l)
    with open(fp, 'w') as o:
        for q, ls in kept.items():
            if ls: o.writelines(ls)
            else: o.write(f"{q}\t{q0[q]}\t*\t*\t*\t*\t*\t*\t*\t*\t*\t0\tmt:f:0\tci:i:1\tsl:i:0\tcm:i:0\tnc:i:0\ts1:i:0\tsm:f:0\n")
    sys.argv = [sys.argv[0], fp, fp, '3', '0.25']
    src = open(os.path.dirname(__file__) + '/triage_sim.py').read().split('print(f"stream:')[0]
    ns = {'__file__': __file__}; exec(compile(src, 'ts', 'exec'), ns)
    names, N, recs, order, qlen = (ns[k] for k in ('names', 'N', 'recs', 'order', 'qlen')); W = ns['W']
    SC = 1.1   # PAF event units per base (measured ~1.1 on KPC2)

    def win_depth(r, sel_mask):
        """estimated per-window depth of read r from overlaps with earlier reads that are in sel_mask"""
        nW = max(1, -(-qlen[r] // W)); diff = np.zeros(nW + 1, np.int32); by_t = defaultdict(list)
        for t, qs, qe in recs[r]:
            if sel_mask is None or sel_mask[order[t]]: by_t[t].append((qs, qe))
        for t, iv in by_t.items():
            iv.sort(); cs, ce = iv[0]
            for s, e in iv[1:] + [(None, None)]:
                if s is not None and s <= ce: ce = max(ce, e); continue
                lo, hi = cs // W, min(nW, -(-ce // W))
                if hi > lo: diff[lo] += 1; diff[hi] -= 1
                if s is not None: cs, ce = s, e
        return np.cumsum(diff)[:nW]

if '2' in PARTS:
    print('\n===== PART 2: overlap-estimated depth vs true same-strand depth (all earlier pooled reads) =====', flush=True)
    BIN = 100; cov = {'+': np.zeros(G // BIN + 2, np.int32), '-': np.zeros(G // BIN + 2, np.int32)}
    rows = []
    for i, r in enumerate(names):
        if r in truth:
            st, a, b = truth[r]
            est = win_depth(r, None)
            for k, ev in enumerate(est):
                pos = (a + (k * W + W / 2) / SC) if st == '+' else (b - (k * W + W / 2) / SC)
                if a <= pos <= b: rows.append((r, real[r], st, int(cov[st][int(pos) // BIN]), int(ev), i))
            cov[st][a // BIN: b // BIN + 1] += 1
    d = pd.DataFrame(rows, columns=['read', 'len', 'strand', 'true_ss', 'est', 'i']); d = d[d.i >= 3000]
    print(f'windows scored: {len(d)} (reads after the first 3000 of the stream)')
    d['tb'] = pd.cut(d.true_ss, [-1, 2, 5, 10, 15, 25, 1000], labels=['0-2', '3-5', '6-10', '11-15', '16-25', '>25'])
    print('\nby TRUE same-strand depth of the window:   windows  mean est  est/true  P(est>=3)  P(est>=10)')
    for k, g in d.groupby('tb', observed=True): print(f'  {k:>6}: {len(g):9d} {g.est.mean():9.1f} {g.est.sum()/max(g.true_ss.sum(),1):9.2f} {(g.est>=3).mean()*100:9.0f}% {(g.est>=10).mean()*100:9.0f}%')
    d['lb'] = pd.cut(d.len, [0, 1000, 3000, 10000, 1e9], labels=['<1kb', '1-3kb', '3-10kb', '>10kb'])
    dd = d[d.true_ss >= 10]
    print('\nwindows with true same-strand depth >=10, by read length:   windows  est/true  P(est>=10)')
    for k, g in dd.groupby('lb', observed=True): print(f'  {k:>7}: {len(g):9d} {g.est.sum()/g.true_ss.sum():9.2f} {(g.est>=10).mean()*100:9.0f}%')
    print('\nmean true same-strand depth in the pooled data (late stream): %.1f ; est: %.1f' % (d.true_ss.mean(), d.est.mean()))

if '3' in PARTS:
    print('\n===== PART 3: stopping rule (triage, D = target estimated same-strand depth) =====', flush=True)
    def triage(D, tau=0.25):
        chosen = np.zeros(N, bool)
        for i, r in enumerate(names):
            e = win_depth(r, chosen)
            if np.maximum(0, D - e).mean() / D >= tau: chosen[i] = True
        return np.where(chosen)[0]
    def genome_cov(sel):
        cv = np.zeros(G // 100 + 2, np.int32)
        for i in sel:
            r = names[i]
            if r in truth: st, a, b = truth[r]; cv[a // 100: b // 100 + 1] += 1
        cv = cv[: G // 100]; return cv
    print(f'{"policy":<30} {"Mb":>5} {"depth":>6} | genome bins >=10x / >=20x / >=30x | 5th-pct depth | SNP F1  indel F1')
    allsel = np.arange(N); cv = genome_cov(allsel)
    r = score([names[i] for i in allsel], 'p3_all')
    print(f'{"ALL reads":<30} {r["Mb"]:5.0f} {r["mean_depth"]:5.1f}x | {(cv>=10).mean()*100:6.0f}% {(cv>=20).mean()*100:6.0f}% {(cv>=30).mean()*100:6.0f}% | {np.percentile(cv,5):10.0f} | {r["SNP"]["F1"]:6.2f} {r["indel"]["F1"]:7.2f}', flush=True)
    for D in (5, 8, 10, 15, 20):
        sel = triage(D); b = sum(real[names[i]] for i in sel); cv = genome_cov(sel)
        r = score([names[i] for i in sel], f'p3_tri{D}')
        print(f'{"TRIAGE D=%d" % D:<30} {r["Mb"]:5.0f} {r["mean_depth"]:5.1f}x | {(cv>=10).mean()*100:6.0f}% {(cv>=20).mean()*100:6.0f}% {(cv>=30).mean()*100:6.0f}% | {np.percentile(cv,5):10.0f} | {r["SNP"]["F1"]:6.2f} {r["indel"]["F1"]:7.2f}', flush=True)
        rr = []
        for k in range(3):
            ids = at_bases(list(rng.permutation(names)), b); rr.append(score(ids, f'p3_tri{D}_rnd{k}'))
        m = mean_r(rr)
        print(f'{"  random, same bases (mean 3)":<30} {m["Mb"]:5.0f} {m["mean_depth"]:5.1f}x | {"":>34} | {m["SNP"]["F1"]:6.2f} {m["indel"]["F1"]:7.2f}', flush=True)
        fc = at_bases(list(names), b); r = score(fc, f'p3_tri{D}_fc')
        print(f'{"  first-come, same bases":<30} {r["Mb"]:5.0f} {r["mean_depth"]:5.1f}x | {"":>34} | {r["SNP"]["F1"]:6.2f} {r["indel"]["F1"]:7.2f}', flush=True)
