"""Stage 1 of the pre-basecalling triage tool: replay of a live run, decisions made causally.

Stream = reads in acquisition order. For each arriving read the engine (rawhash2 --online) has already reported its overlaps with EARLIER reads
(PAF; coordinates in events, reference-free). The triage layer keeps the set S of reads chosen for basecalling and decides for the new read:
    depth_w  = number of reads in S that overlap window w of the new read          (window = 500 events)
    deficit  = mean over windows of max(0, D - depth_w) / D                          (0 = fully covered already, 1 = nothing yet)
    NOW if deficit >= tau (the read is added to S), else BACK BURNER (not basecalled unless budget is left over).
D is the wanted depth in SAME-STRAND reads (Rawsamble only sees same-strand overlaps; total depth is ~2x).

Scoring uses the reference alignments of the reads (never seen by the triage): coverage of the genome by the reads chosen for basecalling.
Comparators at the same number of basecalled bases: first-come (FIFO), random, longest-first (offline), and an oracle that applies the same rule with
exact reference-based depth (both strands, perfect overlaps).

usage: triage_sim.py <online_rawsamble.paf> <truth_ref.paf> [D values] [tau values]
"""
import sys, os
from collections import defaultdict
import numpy as np

rs_paf, truth_paf = sys.argv[1:3]
Ds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 5, 8]
TAUs = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.1, 0.25, 0.5]
W = int(os.environ.get('TRI_W', 500))          # window (events)
CONTIG = os.environ.get('CONTIG')               # score coverage on this reference sequence only (default: every alignment, single-contig references)
rng = np.random.default_rng(0)

# ---- stream: reads in order, with event length and overlaps with earlier reads
order, qlen, recs = {}, {}, defaultdict(list)
for line in open(rs_paf):
    c = line.rstrip("\n").split("\t")
    q = c[0]
    if q not in order: order[q] = len(order); qlen[q] = int(c[1])
    if c[5] != "*": recs[q].append((c[5], int(c[2]), int(c[3])))
names = sorted(order, key=order.get); N = len(names)

# ---- truth (scoring only): primary alignment of each read on the reference
aln, reflen, blen = {}, 0, {}
for line in open(truth_paf):
    c = line.split("\t")
    if c[0] in aln or c[0] in blen: continue
    blen[c[0]] = int(c[1])
    if CONTIG and c[5] != CONTIG: continue
    reflen = int(c[6])
    if int(c[11]) >= 20: aln[c[0]] = (int(c[7]), int(c[8]), c[4])
bases = np.array([blen.get(r, 0) for r in names], dtype=np.int64)


def coverage(sel):
    d = np.zeros(reflen + 1, np.int32)
    for i in sel:
        a = aln.get(names[i])
        if a: d[a[0]] += 1; d[a[1]] -= 1
    return np.cumsum(d)[:reflen]


def report(label, sel):
    cov = coverage(sel); b = bases[list(sel)].sum()
    return (f"{label:<34} reads {100*len(sel)/N:5.1f}%  bases {100*b/bases.sum():5.1f}%  "
            f"mean depth {cov.mean():5.1f}  >=1x {100*(cov>=1).mean():5.1f}%  >=5x {100*(cov>=5).mean():5.1f}%  >=10x {100*(cov>=10).mean():5.1f}%  "
            f"<5x {100*(cov<5).mean():5.1f}%"), b


def run_triage(D, tau):
    chosen = np.zeros(N, bool); sel = []
    for i, r in enumerate(names):
        nW = max(1, -(-qlen[r] // W))
        diff = np.zeros(nW + 1, np.int32)
        by_target = defaultdict(list)
        for t, qs, qe in recs[r]:
            if chosen[order[t]]: by_target[t].append((qs, qe))
        for t, iv in by_target.items():                       # union of this target's chains, so one read counts once per window
            iv.sort(); cs, ce = iv[0]
            for s, e in iv[1:] + [(None, None)]:
                if s is not None and s <= ce: ce = max(ce, e); continue
                lo, hi = cs // W, min(nW, -(-ce // W))
                if hi > lo: diff[lo] += 1; diff[hi] -= 1
                if s is not None: cs, ce = s, e
        depth = np.cumsum(diff)[:nW]
        deficit = np.maximum(0, D - depth).mean() / D
        if deficit >= tau: chosen[i] = True; sel.append(i)
    return sel


def run_oracle(D, tau):
    """same rule, exact depth from the reference (both strands: wanted total depth 2D), perfect overlaps"""
    cov = np.zeros(reflen + 1, np.int32); sel = []
    cum = None
    # incremental coverage via a Fenwick-free approach: recompute window depth from a running difference array
    diff = np.zeros(reflen + 1, np.int32)
    run = np.zeros(reflen, np.int32)
    for i, r in enumerate(names):
        a = aln.get(r)
        if not a: continue
        s, e, _ = a
        seg = run[s:e]
        if len(seg) == 0: continue
        deficit = np.maximum(0, 2 * D - seg).mean() / (2 * D)
        if deficit >= tau:
            run[s:e] += 1; sel.append(i)
    return sel


def budget_baselines(b_target):
    out = {}
    cum = np.cumsum(bases); k = int(np.searchsorted(cum, b_target)) + 1
    out["first-come (FIFO)"] = list(range(k))
    perm = rng.permutation(N); cs = np.cumsum(bases[perm]); kk = int(np.searchsorted(cs, b_target)) + 1
    out["random"] = list(perm[:kk])
    desc = np.argsort(-bases); cs = np.cumsum(bases[desc]); kk = int(np.searchsorted(cs, b_target)) + 1
    out["longest-first (offline)"] = list(desc[:kk])
    return out


print(f"stream: {N} reads, {bases.sum()/1e6:.1f} Mb basecalled if everything is basecalled, genome {reflen/1e6:.2f} Mb (all reads: mean depth {bases.sum()/reflen:.1f}x)")
line, _ = report("basecall everything", list(range(N)))
print(line)
for D in Ds:
    for tau in TAUs:
        sel = run_triage(D, tau)
        line, b = report(f"TRIAGE D={D} tau={tau}", sel)
        print(line)
        if tau == TAUs[len(TAUs) // 2]:                                   # baselines at equal bases, once per D
            for name, s in budget_baselines(b).items():
                print(report(f"   same bases: {name}", s)[0])
            print(report(f"   oracle (exact depth) D={D}", run_oracle(D, tau))[0])
