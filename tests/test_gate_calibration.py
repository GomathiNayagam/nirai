import numpy as np
from nirai.gate import CalibratedGate


def run(scores, **kw):
    g = CalibratedGate(**kw); out = []
    for i, s in enumerate(scores): out += g.push(f"r{i}", s)
    return out + g.flush()


def test_warmup_holds_then_releases_all():
    rng = np.random.default_rng(0); s = rng.normal(0.8, 0.05, 1000)
    g = CalibratedGate(keep=0.98, warmup=300); first = []
    for i in range(299): first += g.push(f"r{i}", s[i])
    assert first == []                                  # still waiting
    out = g.push("r299", s[299]); assert len(out) == 300 and out[0][0] == "r0"
    assert 0.95 <= np.mean([d == "basecall" for _, _, d in out]) <= 1.0


def test_keeps_about_98_percent_and_follows_drift():
    rng = np.random.default_rng(1); s = np.concatenate([rng.normal(0.8, 0.05, 3000), rng.normal(0.7, 0.05, 3000)])   # run drifts down
    kept = lambda o: np.mean([d == "basecall" for _, _, d in o])
    assert kept(run(s, keep=0.98, rolling=True)[3300:]) > 0.97       # rolling follows the drift
    assert kept(run(s, keep=0.98, rolling=False)[3300:]) < 0.5       # fixed threshold drops most reads after the drift


def test_nan_always_basecalled_and_short_run():
    out = run([np.nan, 0.9, 0.1], keep=0.98, warmup=300); assert len(out) == 3 and out[0][2] == "basecall"
