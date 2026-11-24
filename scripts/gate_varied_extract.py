"""Pull the first 5000 pA samples (1.0 s) of every read from a downloaded pod5 set, with a Q label and an arrival rank, into <dir>/prefix5000.npz.
usage: gate_varied_extract.py DIR [--summary full_ss.txt]   (species dirs: labels from DIR/hac.fastq public hac calls; HG002: labels from the MinKNOW sequencing summary mean_qscore_template)
Reads with < 5000 samples are dropped; at most MAXN reads (the earliest by start sample) are kept."""
import os, sys, glob, numpy as np, pandas as pd, pod5
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0]); from nirai.features import read_pa
NP, MAXN = 5000, 80000
def work(f):
    X, rid, st = [], [], []; isfail = os.path.basename(f).split('__')[0].endswith('_fail')
    with pod5.Reader(f) as r:
        for rec in r.reads():
            x = read_pa(rec)
            if len(x) >= NP: X.append(x[:NP].astype(np.float16)); rid.append(str(rec.read_id) + (':F' if isfail else ':P')); st.append(int(rec.start_sample))
    return (np.stack(X) if X else np.zeros((0, NP), np.float16)), rid, st
if __name__ == '__main__':
    D = sys.argv[1].rstrip('/') + '/'; files = sorted(glob.glob(D + 'raw/**/*.pod5', recursive=True)); print(len(files), 'pod5 files', flush=True)
    with get_context('fork').Pool(8) as p: parts = list(p.imap(work, files))
    X = np.concatenate([a for a, _, _ in parts]); rid = [r for _, b, _ in parts for r in b]; st = np.array([s for _, _, c in parts for s in c])
    flag = np.array([r[-1] for r in rid]); rid = np.array([r[:-2] for r in rid]); q = {}
    if '--folder' in sys.argv:       # label from the MinKNOW pass/fail folder (fail = below ~Q9): proxy Q 5 / 15
        q = dict(zip(rid, np.where(flag == 'F', 5.0, 15.0)))
    elif '--summary' in sys.argv:
        s = pd.read_csv(D + sys.argv[sys.argv.index('--summary') + 1], sep='\t', usecols=['read_id', 'mean_qscore_template']); q = dict(zip(s.read_id, s.mean_qscore_template))
    else:
        with open(D + 'hac.fastq') as f:
            while True:
                h = f.readline()
                if not h: break
                sq = f.readline().strip(); f.readline(); ql = f.readline().strip()
                if len(sq) == len(ql) and ql: a = np.frombuffer(ql.encode(), np.uint8).astype(float) - 33; q[h[1:].split()[0]] = -10 * np.log10(np.mean(10 ** (-a / 10)))
    Q = np.array([q.get(r, np.nan) for r in rid]); rid = np.array(rid); o = np.argsort(st, kind='stable')[:MAXN]; o = o[np.argsort(st[o], kind='stable')]
    np.savez(D + 'prefix5000.npz', X=X[o], Q=Q[o], RID=rid[o], I=np.arange(len(o)))
    Qo = Q[o]; print(f'{D}: {len(o)} reads >= 1 s kept (of {len(rid)}); labelled {int((~np.isnan(Qo)).sum())}; low (Q<10) {int((Qo < 10).sum())} = {100 * np.nanmean(Qo < 10):.1f}% of labelled')
