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


def test_short_reads_always_basecalled_and_not_counted_as_redundant():
    names, qlen, recs = _paf_recs(10, length=900)           # 900 events ~ 820 bases: below the 1 kb policy
    assert triage(names, qlen, recs, D=3).all()
    assert not triage(names, qlen, recs, D=3, min_bases=0).all()


def test_length_correction_stops_earlier_because_missed_overlaps_are_credited():
    names, qlen, recs = _paf_recs(40, length=3000)          # ~2.7 kb reads: recall 0.5, so 3 seen overlaps count as depth 6 >= D=5
    plain = triage(names, qlen, recs, D=5, length_correct=False).sum(); corr = triage(names, qlen, recs, D=5).sum()
    assert corr < plain


def test_short_reads_can_be_dropped():
    names, qlen, recs = _paf_recs(10, length=900)
    assert not triage(names, qlen, recs, D=3, short_reads="drop").any()
    names, qlen, recs = _paf_recs(10, length=3000); qlen["r4"] = 500
    ch = triage(names, qlen, recs, D=3, short_reads="drop"); assert not ch[4]


def test_groups_bring_each_barcode_to_depth_separately():
    names, qlen, recs = _paf_recs(20)                       # all 20 reads overlap (same genome)
    g = ["bc1" if i % 2 == 0 else "bc2" for i in range(20)]
    ch = triage(names, qlen, recs, D=3, groups=g)
    assert ch[::2].sum() == 3 and ch[1::2].sum() == 3       # 3 reads per barcode instead of 3 in total
    assert triage(names, qlen, recs, D=3).sum() == 3
