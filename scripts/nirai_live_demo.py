"""Live demo of the Nirai gate. Reads arrive one at a time in random order (as from a flow cell). For each read only the first 0.4 s of signal
is used. Labels (from labels.tsv) are shown AFTER the decision, for checking only; the gate never sees them.
usage: nirai_live_demo.py MODEL POD5 LABELS [--n 300] [--rate 20] [--seed 1]"""
import sys, time, argparse, random
import numpy as np, pandas as pd, pod5
sys.path.insert(0, __file__.rsplit('/scripts', 1)[0])
from nirai.gate import load_model
from nirai.features import PREFIXES, prefix_features, read_pa
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('pod5'); ap.add_argument('labels')
ap.add_argument('--n', type=int, default=300); ap.add_argument('--rate', type=float, default=20, help='reads per second arriving (0 = as fast as possible)'); ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
m = load_model(a.model); npre = PREFIXES[m['prefix']]; thr = m['threshold']; lab = pd.read_csv(a.labels, sep='\t').set_index('read_id').label.to_dict()
G, R, X = '\033[32m', '\033[31m', '\033[0m'
with pod5.Reader(a.pod5) as r:
    ids = [str(x) for x in r.read_ids]; random.Random(a.seed).shuffle(ids); ids = ids[:a.n]
    tp = fp = tn = fn = 0; lat = []; t_start = time.time()
    print(f'{"time":>6} {"read":<10} {"score":>6}  {"decision":<9} {"gate ms":>8}   (true label, shown after)')
    for k, rec in enumerate(r.reads(selection=ids)):
        due = t_start + k / a.rate if a.rate > 0 else 0
        if a.rate > 0 and time.time() < due: time.sleep(due - time.time())
        t0 = time.perf_counter()                                   # read's first 0.4 s of signal is available: decide
        ft = prefix_features(read_pa(rec), npre)
        s = float(m['model'].predict_proba(pd.DataFrame([ft], columns=m['features']))[0, 1]) if ft is not None else float('nan')
        dec = 'basecall' if (ft is None or s >= thr) else 'SKIP'
        ms = (time.perf_counter() - t0) * 1000; lat.append(ms); y = lab[str(rec.read_id)]
        ok = (dec == 'SKIP') == (y == 'fail')
        tp += dec == 'SKIP' and y == 'fail'; fp += dec == 'SKIP' and y == 'pass'; tn += dec != 'SKIP' and y == 'pass'; fn += dec != 'SKIP' and y == 'fail'
        c = G if dec != 'SKIP' else R
        print(f'{time.time()-t_start:6.1f} {str(rec.read_id)[:8]:<10} {s:6.3f}  {c}{dec:<9}{X} {ms:8.2f}   {y:<5} {"ok" if ok else "WRONG"}', flush=True)
        if (k + 1) % 50 == 0:
            print(f'   -- after {k+1} reads: fail reads skipped {tp}/{tp+fn} ({100*tp/max(tp+fn,1):.0f}%), pass reads wrongly skipped {fp}/{fp+tn} ({100*fp/max(fp+tn,1):.0f}%) --', flush=True)
lat = np.array(lat)
print(f'\n{len(lat)} reads. Gate compute per read: median {np.median(lat):.2f} ms, 95th pct {np.percentile(lat,95):.2f} ms, max {lat.max():.2f} ms')
print(f'Decision is available 0.4 s after a read starts (the signal it needs) + {np.median(lat):.1f} ms compute.')
print(f'fail reads skipped {tp}/{tp+fn}; pass reads kept {tn}/{tn+fp}')
