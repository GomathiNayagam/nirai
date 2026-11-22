import numpy as np
from nirai.triage import triage


def _paf_recs(n_reads, length=3000, overlap_every=1):
    names = [f"r{i}" for i in range(n_reads)]
    qlen = {r: length for r in names}
    recs = {r: [] for r in names}
    for i in range(1, n_reads):                     # every read fully overlaps all earlier reads
        recs[names[i]] = [(names[j], 0, length) for j in range(i)]
    return names, qlen, recs


def test_redundant_reads_are_skipped_once_depth_is_reached():
    names, qlen, recs = _paf_recs(20)
    ch = triage(names, qlen, recs, D=3, tau=0.25, window=1000)
    assert ch[:3].all() and not ch[3:].any()        # first three reads fill depth 3, the rest are redundant


def test_no_overlaps_means_everything_is_chosen():
    names = [f"r{i}" for i in range(5)]
    ch = triage(names, {r: 3000 for r in names}, {r: [] for r in names}, D=3)
    assert ch.all()


def test_eligibility_mask_removes_reads():
    names = [f"r{i}" for i in range(4)]
    ch = triage(names, {r: 3000 for r in names}, {r: [] for r in names}, D=3, eligible=np.array([True, False, True, False]))
    assert ch.tolist() == [True, False, True, False]


def test_decision_is_causal():
    names, qlen, recs = _paf_recs(12)
    full = triage(names, qlen, recs, D=3)
    cut = triage(names[:8], qlen, {r: [o for o in recs[r] if o[0] in set(names[:8])] for r in names[:8]}, D=3)
    assert (full[:8] == cut).all()
