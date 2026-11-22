"""Signal-only features of a read (or of its first seconds). No basecalls, no reference."""
import numpy as np

SAMPLE_RATE = 5000                      # R10.4.1 default (Hz)
FEATURES = ("sd", "mad_diff", "frac_hi")    # median pA is deliberately left out: it depends on the sample and did not transfer across runs
PREFIXES = {"0.4s": 2000, "0.8s": 4000, "2s": 10000, "4s": 20000}   # samples at 5 kHz
MIN_SAMPLES = 300


def signal_features(x):
    """sd, MAD of sample-to-sample differences, fraction of samples further than 4 sd from the median."""
    d = np.diff(x)
    mad = float(np.median(np.abs(d - np.median(d))))
    med = float(np.median(x))
    sd = float(x.std())
    return [sd, mad, float((np.abs(x - med) > 4 * sd).mean())]


def read_pa(rec):
    """Picoampere-scaled signal of a pod5 ReadRecord (scale only, same convention as the experiments)."""
    return rec.signal.astype(np.float32) * rec.calibration.scale


def prefix_features(x, n_prefix):
    """Features of the first n_prefix samples; None if the read is still shorter than the prefix."""
    if len(x) < max(n_prefix, MIN_SAMPLES):
        return None
    return signal_features(x[:n_prefix])
