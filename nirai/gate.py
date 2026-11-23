"""Pass/fail gate: predict from the first samples of a read whether it will be a low-quality ('fail') read, before basecalling.
Train on reads whose pass/fail label is known (ONT pass/fail folders), then score new POD5 files."""
import glob, os
import numpy as np
import pandas as pd
import pod5
from .features import PREFIXES, FEATURES, prefix_features, read_pa


def _pod5_files(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.pod5"))) if os.path.isdir(p) else [p]
    return out


def feature_table(paths, prefix="0.4s", label=None, max_reads=None):
    """rows: (file, read_id, label, n_samples, features...). Reads shorter than the prefix are kept with NaN features. max_reads is per file."""
    n_prefix = PREFIXES[prefix]; rows = []
    for f in _pod5_files(paths):
        try:
            r = pod5.Reader(f)
        except Exception as e:                     # damaged file: skip, keep going
            print(f"skip {f}: {e}"); continue
        with r:
            n = 0
            for rec in r.reads():
                ft = prefix_features(read_pa(rec), n_prefix) or [np.nan] * len(FEATURES)
                rows.append([os.path.basename(f), str(rec.read_id), label, int(rec.sample_count)] + ft)
                n += 1
                if max_reads and n >= max_reads: break      # per file
    return rows


COLS = ["file", "read_id", "label", "n_samples"] + list(FEATURES)


def train(pass_paths, fail_paths, prefix="0.4s", pass_recall=0.95, max_reads=None):
    """Fit a gradient-boosting classifier (1 = pass). Threshold is set so that `pass_recall` of held-out pass reads survive
    (grouped CV by file, so no file is in both train and test). Returns a dict that save_model() can store."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame(feature_table(pass_paths, prefix, 1, max_reads) + feature_table(fail_paths, prefix, 0, max_reads), columns=COLS)
    d = d.dropna(subset=list(FEATURES)); X, y, g = d[list(FEATURES)], d["label"].values, d["file"].values
    nfold = min(5, len(set(g)))
    if nfold < 2: raise ValueError("need at least 2 distinct POD5 files for grouped cross-validation")
    mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(nfold).split(X, y, g):
        p[te] = mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    thr = float(np.quantile(p[y == 1], 1 - pass_recall))
    cv = {"auc": float(roc_auc_score(y, p)), "fail_rejected": float((p[y == 0] < thr).mean()), "pass_kept": float((p[y == 1] >= thr).mean()),
          "n_pass": int((y == 1).sum()), "n_fail": int((y == 0).sum()), "n_files": int(len(set(g)))}
    return {"model": mk().fit(X, y), "prefix": prefix, "threshold": thr, "features": list(FEATURES), "cv": cv}


def save_model(m, path):
    import joblib, sklearn
    m = dict(m); m["sklearn"] = sklearn.__version__; joblib.dump(m, path)


def load_model(path):
    import joblib
    return joblib.load(path)


def score_pod5(model, paths, threshold=None):
    """yield (read_id, score, decision). decision 'basecall' if score >= threshold, 'skip' otherwise, 'basecall' (undecided) if the read is
    shorter than the prefix (too little signal to judge)."""
    thr = model["threshold"] if threshold is None else threshold; n_prefix = PREFIXES[model["prefix"]]
    for f in _pod5_files(paths):
        with pod5.Reader(f) as r:
            for rec in r.reads():
                ft = prefix_features(read_pa(rec), n_prefix)
                if ft is None: yield str(rec.read_id), float("nan"), "basecall"; continue
                s = float(model["model"].predict_proba(pd.DataFrame([ft], columns=model["features"]))[0, 1])
                yield str(rec.read_id), s, ("basecall" if s >= thr else "skip")


class CalibratedGate:
    """Streaming, label-free threshold for a pass/fail gate. Scores are P(pass): a read is skipped when its score is below the threshold.
    The threshold is the (1 - keep) quantile of recent scores, so about `keep` of reads like the recent ones survive (good reads dominate, so this tracks good-read retention).
      warm-up : the first `warmup` reads are scored but undecided ('hold'); once they are in, their decisions are released with the first threshold (no basecalling needed to wait).
      rolling : after warm-up the threshold is recomputed every `update` reads from the last `window` scores (follows slow drift within a run).
    No labels and no basecalls are used. Reads too short to score (score NaN) are always 'basecall'."""
    def __init__(self, keep=0.98, warmup=300, window=1000, update=100, rolling=True):
        self.keep, self.warmup, self.window, self.update, self.rolling = keep, warmup, window, update, rolling
        self.scores = []; self.thr = None; self._n_at_update = 0; self.held = []

    def _recompute(self):
        recent = np.array(self.scores[-self.window:]); recent = recent[~np.isnan(recent)]
        self.thr = float(np.quantile(recent, 1 - self.keep)) if len(recent) else None; self._n_at_update = len(self.scores)

    def _decide(self, s):
        return "basecall" if (np.isnan(s) or s >= self.thr) else "skip"

    def push(self, read_id, score):
        """add one read's score. Returns a list of (read_id, score, decision) now final: empty during warm-up, the whole warm-up batch when it ends, then one per read."""
        self.scores.append(score)
        if self.thr is None:
            self.held.append((read_id, score))
            if len([x for x in self.scores if not np.isnan(x)]) < self.warmup: return []
            self._recompute(); out = [(r, s, self._decide(s)) for r, s in self.held]; self.held = []; return out
        if self.rolling and len(self.scores) - self._n_at_update >= self.update: self._recompute()
        return [(read_id, score, self._decide(score))]

    def flush(self):
        """end of run with fewer reads than the warm-up: decide the held reads with what was seen (or basecall them all if nothing could be scored)."""
        if self.thr is None: self._recompute()
        out = [(r, s, "basecall" if self.thr is None else self._decide(s)) for r, s in self.held]; self.held = []; return out
