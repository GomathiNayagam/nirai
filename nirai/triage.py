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


def triage(names, qlen, recs, D=5, tau=0.25, window=1000, eligible=None):
    """-> boolean array chosen[i]. `eligible` (bool array) lets a pass/fail gate remove reads before the redundancy rule."""
    idx_of = {r: i for i, r in enumerate(names)}; chosen = np.zeros(len(names), bool)
    for i, r in enumerate(names):
        if eligible is not None and not eligible[i]: continue
        e = window_depth(qlen[r], recs[r], chosen, idx_of, window)
        if np.maximum(0, D - e).mean() / D >= tau: chosen[i] = True
    return chosen
