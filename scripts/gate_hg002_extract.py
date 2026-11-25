"""HG002 PromethION test, step 1: pull the first 5000 pA samples (1.0 s) of up to PER_FILE reads from every downloaded pod5 file of one run.
usage: gate_hg002_extract.py RUN_DIR      (RUN_DIR has pass/ and fail/ folders of pod5 files; label = the folder, fail=1)
Writes RUN_DIR/prefix5000.npz: X (float16), FAIL (0/1), FILE (file name), ST (start sample).  Reads shorter than 5000 samples are dropped."""
import os, sys, glob, numpy as np, pod5
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]); from nirai.features import read_pa
NP, PER_FILE = 5000, 4000
def work(f):
    X, st = [], []; fail = int(f.split('/')[-2].endswith('fail'))
    with pod5.Reader(f) as r:
        for rec in r.reads():
            if len(X) >= PER_FILE: break
            x = read_pa(rec)
            if len(x) >= NP: X.append(x[:NP].astype(np.float16)); st.append(int(rec.start_sample))
    return (np.stack(X) if X else np.zeros((0, NP), np.float16)), np.full(len(X), fail), np.full(len(X), os.path.basename(f)), np.array(st)
if __name__ == '__main__':
    D = sys.argv[1].rstrip('/') + '/'; files = sorted(glob.glob(D + '*/*.pod5')); print(D, len(files), 'files', flush=True)
    with get_context('fork').Pool(6) as p: parts = list(p.imap(work, files))
    X = np.concatenate([a for a, _, _, _ in parts]); F = np.concatenate([b for _, b, _, _ in parts]); N = np.concatenate([c for _, _, c, _ in parts]); S = np.concatenate([d for _, _, _, d in parts])
    np.savez(D + 'prefix5000.npz', X=X, FAIL=F, FILE=N, ST=S); print(f'{D}: {len(F)} reads, fail {int(F.sum())} ({100 * F.mean():.1f}%)', flush=True)
