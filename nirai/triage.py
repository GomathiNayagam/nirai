"""Causal redundancy triage from raw-signal overlaps (Rawsamble / RawHash2 --online PAF). Reads must be listed in arrival order;
each overlap points to an EARLIER read. The rule uses only overlap records and signal lengths (no reference, no basecalls).

    depth_w = number of already-chosen reads overlapping window w of the new read
    deficit = mean_w max(0, D - depth_w) / D
    choose  = deficit >= tau
"""
from collections import defaultdict
import numpy as np


def read_paf(path):
    """-> names (arrival order), qlen {read: length in events}, recs {read: [(target, qstart, qend)]}"""
    order, qlen, recs = {}, {}, defaultdict(list)
    with open(path) as f:
        for line in f:
            c = line.split("\t", 6)
            q = c[0]
            if q not in order: order[q] = len(order); qlen[q] = int(c[1])
            if c[5] != "*": recs[q].append((c[5], int(c[2]), int(c[3])))
    return sorted(order, key=order.get), qlen, recs


def window_depth(qlen_r, overlaps, chosen_idx, idx_of, window):
    """per-window depth of one read from overlaps with already-chosen earlier reads (each target counted once per window)"""
    nW = max(1, -(-qlen_r // window)); diff = np.zeros(nW + 1, np.int32); by_t = defaultdict(list)
    for t, qs, qe in overlaps:
        j = idx_of.get(t)
        if j is not None and chosen_idx[j]: by_t[t].append((qs, qe))
    for iv in by_t.values():
        iv.sort(); cs, ce = iv[0]
        for s, e in iv[1:] + [(None, None)]:
            if s is not None and s <= ce: ce = max(ce, e); continue
            lo, hi = cs // window, min(nW, -(-ce // window))
            if hi > lo: diff[lo] += 1; diff[hi] -= 1
            if s is not None: cs, ce = s, e
    return np.cumsum(diff)[:nW]


EVENTS_PER_BASE = 1.1          # PAF event units per base (measured on KPC2 and RDH275)
# estimated/true depth by read length, measured on KPC2 (frozen; RDH275 gave 0.05/0.31/0.57/0.70/0.74): <1, 1-2, 2-4, 4-8, >8 kb. Short overlaps are missed, so long reads look less covered than they are.
RECALL_BY_LEN = ((1000, 0.2), (2000, 0.21), (4000, 0.50), (8000, 0.63), (float("inf"), 0.76))


def length_recall(n_bases):
    for hi, r in RECALL_BY_LEN:
        if n_bases < hi: return r


def triage(names, qlen, recs, D=5, tau=0.25, window=1000, eligible=None, min_bases=1000, length_correct=None, short_reads="basecall", groups=None):
    """-> boolean array chosen[i]. `eligible` (bool array) lets a pass/fail gate remove reads before the redundancy rule.
    min_bases: reads shorter than this (estimated from signal length) are always chosen for basecalling and take no part in the decision (they are cheap to keep and overlaps are rarely found for them). 0 disables.
    short_reads: 'basecall' (default) keeps reads under min_bases; 'drop' never selects them (whole-genome work where such reads add little; not for amplicons or small fragments).
    groups: optional per-read group labels (e.g. native barcode). Depth is then counted only from already-chosen reads of the SAME group, so every barcode is brought to depth D on its own (without it, a barcode whose genome is already covered by another barcode would be skipped as redundant).
    length_correct: divide the estimated depth by the length-dependent overlap recall (capped at D). None = on when D >= 5 (it helped at D=5 and 10 and hurt at D=3 in replays)."""
    if length_correct is None: length_correct = D >= 5
    idx_of = {r: i for i, r in enumerate(names)}; chosen = np.zeros(len(names), bool)
    grp = None if groups is None else np.asarray(groups)
    for i, r in enumerate(names):
        if eligible is not None and not eligible[i]: continue
        nb = qlen[r] / EVENTS_PER_BASE
        if min_bases and nb < min_bases:
            chosen[i] = short_reads == "basecall"; continue
        e = window_depth(qlen[r], recs[r], chosen if grp is None else chosen & (grp == grp[i]), idx_of, window)
        if length_correct: e = np.minimum(e / length_recall(nb), D)
        if np.maximum(0, D - e).mean() / D >= tau: chosen[i] = True
    return chosen
