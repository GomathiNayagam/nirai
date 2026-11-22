# Does an assembly graph built only from Rawsamble raw-signal overlaps (no basecalls) behave like one built from basecalled reads?
# Arms (all via miniasm layout; same reads, same stream order, subsets of first N):
#   SIGNAL  : Rawsamble online.paf (same-strand overlaps only), dummy sequences (layout only)
#   BC_FWD  : minimap2 -x ava-ont --for-only on Dorado hac reads (same-strand only; isolates overlap sensitivity)
#   BC_BOTH : minimap2 -x ava-ont, both strands (normal assembly)
# Scored against reference positions of reads (truth_sig.paf): genome covered by unitig reads, unitig spans, chimeric unitigs.
import sys, os, subprocess, collections, numpy as np
base = os.environ.get('KROOT', '/home/sgomathi/rawsamble_work/kpc2_run').rstrip('/') + '/'
W = base + 'asm/'; os.makedirs(W, exist_ok=True)
Ns = [int(x) for x in sys.argv[1].split(',')]
MINLEN = int(os.environ.get('MINLEN', '0'))

order = [l.rstrip('\n').split('\t') for l in open(base + 'stream/order.tsv')][1:]
nsamp = {o[0]: int(o[2]) for o in order}
# basecalled read ids / sequences in stream order
seqs = {}
with open(base + 'stream/first8000_hac.fastq') as f:
    while True:
        h = f.readline()
        if not h: break
        s = f.readline().strip(); f.readline(); f.readline()
        seqs[h[1:].split()[0]] = s
ids_all = [o[0] for o in order if o[0] in seqs]
pos = {r: i for i, r in enumerate(ids_all)}
truth = {}
tlen = {}
for l in open(base + 'run/truth_sig.paf'):
    f = l.rstrip('\n').split('\t')
    if f[5] == '*': continue
    truth[f[0]] = (f[5], int(f[7]), int(f[8]))
    tlen[f[5]] = int(f[6])
G = sum(tlen.values())

# signal paf: scale so that event units are consistent with n_samples*0.08
qr = []
for l in open(base + 'run/online.paf'):
    f = l.split('\t')
    if f[5] == '*' or f[0] not in nsamp: continue
    qr.append(int(f[1]) / (nsamp[f[0]] * 0.08))
S = float(np.median(qr))
print(f'signal PAF unit scale vs n_samples*0.08: {S:.3f}')

LEN = {}   # per-read length in event units: at least n_samples*0.08*S and at least the largest coordinate in any overlap line
def filt_signal(Nset, out):
    lines = []
    L = {r: int(round(nsamp[r] * 0.08 * S)) for r in Nset}
    for l in open(base + 'run/online.paf'):
        f = l.split('\t')
        if f[5] == '*' or f[0] not in Nset or f[5] not in Nset or f[0] == f[5]: continue
        lines.append(f)
        L[f[0]] = max(L[f[0]], int(f[3])); L[f[5]] = max(L[f[5]], int(f[8]))
    with open(out, 'w') as o:
        for f in lines:
            f[1] = str(L[f[0]]); f[6] = str(L[f[5]])
            o.write('\t'.join(f[:12]) + '\n')
    LEN.clear(); LEN.update(L)
    return len(lines)

def filt_mm(src, Nset, out):
    n = 0
    with open(out, 'w') as o:
        for l in open(src):
            f = l.split('\t')
            if f[0] in Nset and f[5] in Nset and f[0] != f[5]:
                o.write(l); n += 1
    return n

def write_reads(Nset, out, dummy):
    with open(out, 'w') as o:
        for r in ids_all:
            if r in Nset:
                s = 'A' * LEN.get(r, int(round(nsamp[r] * 0.08 * S))) if dummy else seqs[r]
                o.write('>%s\n%s\n' % (r, s))

def run_miniasm(reads, paf, gfa):
    with open(gfa, 'w') as o:
        subprocess.run(['miniasm', '-f', reads, paf], stdout=o, stderr=subprocess.DEVNULL, check=True)
    utg = collections.defaultdict(list)
    for l in open(gfa):
        f = l.rstrip('\n').split('\t')
        if f[0] == 'a':   # a  utg  offset  read  strand  len
            utg[f[1]].append(f[3].rsplit(':', 1)[0])
    return utg

def union_len(iv):
    iv = sorted(iv); tot = 0; cs, ce = None, None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return tot + (ce - cs if cs is not None else 0)

def components(iv):
    iv = sorted(iv); comps = []
    for s, e in iv:
        if comps and s <= comps[-1][1] + 5000: comps[-1][1] = max(comps[-1][1], e)
        else: comps.append([s, e])
    return comps

def score(utg, Nset):
    cover = collections.defaultdict(list); spans = []; chim = 0; nu = 0; in_u = 0
    for u, rs in utg.items():
        if len(rs) < 2: continue
        nu += 1; in_u += len(rs)
        per = collections.defaultdict(list)
        for r in rs:
            if r in truth:
                t, a, b = truth[r]; per[t].append((a, b)); cover[t].append((a, b))
        comps = sum(len(components(v)) for v in per.values())
        if comps > 1: chim += 1
        spans.append(sum(union_len(v) for v in per.values()))
    cov = sum(union_len(v) for v in cover.values()) / G
    spans = sorted(spans, reverse=True)
    tot = sum(spans); n50 = 0; c = 0
    for s in spans:
        c += s
        if c >= G / 2: n50 = s; break           # NG50 against genome size
    return dict(unitigs=nu, reads_in_unitigs=in_u / len(Nset), genome_cov=cov, longest=(spans[0] if spans else 0), NG50=n50, chimeric=chim / max(nu, 1))

print(f'{"N":>5} {"depth":>6} {"arm":8} {"overlaps":>9} {"unitigs":>7} {"reads in":>8} {"genome":>7} {"longest":>8} {"NG50":>8} {"chimeric":>8}')
for N in Ns:
    ids = ids_all[:N]; Nset = set(ids)
    depth = sum(len(seqs[r]) for r in ids) / G
    for arm in ('SIGNAL', 'BC_FWD', 'BC_BOTH'):
        paf = W + f'{arm}_{N}.paf'; rd = W + f'{arm}_{N}.fa'; gfa = W + f'{arm}_{N}.gfa'
        if arm == 'SIGNAL': n = filt_signal(Nset, paf); write_reads(Nset, rd, True)
        else: n = filt_mm(W + ('ava_for.paf' if arm == 'BC_FWD' else 'ava_both.paf'), Nset, paf); write_reads(Nset, rd, False)
        utg = run_miniasm(rd, paf, gfa); m = score(utg, Nset)
        print(f'{N:>5} {depth:5.1f}x {arm:8} {n:9d} {m["unitigs"]:7d} {m["reads_in_unitigs"]*100:7.0f}% {m["genome_cov"]*100:6.0f}% {m["longest"]/1000:7.0f}k {m["NG50"]/1000:7.0f}k {m["chimeric"]*100:7.0f}%', flush=True)
