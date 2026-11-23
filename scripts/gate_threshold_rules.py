"""Compare label-free threshold rules on saved gate scores (gate_calibrated_threshold.py writes scores_<run>.npz: p = P(low), y = low label, runs in stream order).
Rules (all use only the scores of the first K reads, or the last 1000 for rolling; no labels):
  pct   : q-th percentile of ALL scores            (breaks when a run has many low reads: they sit in the upper tail)
  robust: logit-median + z * 1.4826 * MAD          (z = 2.05 ~ 98% of good reads if good reads are roughly normal in logit space; low reads barely move median/MAD)
usage: gate_threshold_rules.py [DIR]"""
import sys, os, numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/rawsamble_work/gate_loro/failaug/')
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def thr_pct(s, q): return np.percentile(s, q)
def thr_rob(s, z): l = lg(s); m = np.median(l); mad = np.median(np.abs(l - m)) * 1.4826; return 1 / (1 + np.exp(-(m + z * mad)))
print(f'{"run":<18}{"K":>5}  {"rule":<14}{"good kept":>10}{"low caught":>11}   rolling(last 1000, /100 reads): good kept / low caught')
for run in ['RDH275', 'KPC2_FAX39616', 'KPC2_FAX76295', 'ECOLI_ATCC25922']:
    z = np.load(D + f'scores_{run}.npz'); p, y = z['p'], z['y']
    for K in (300, 1000):
        ev, ye = p[K:], y[K:]
        for name, f, arg in (('pct 98', thr_pct, 98), ('robust z=2.05', thr_rob, 2.05), ('robust z=2.33', thr_rob, 2.33)):
            thr = f(p[:K], arg); gk = (ev[ye == 0] <= thr).mean(); lc = (ev[ye == 1] > thr).mean()
            tr = np.array([f(p[max(0, K + b - 1000):K + b], arg) for b in range(0, len(ev), 100)]).repeat(100)[:len(ev)]
            rg = (ev[ye == 0] <= tr[ye == 0]).mean(); rl = (ev[ye == 1] > tr[ye == 1]).mean()
            print(f'{run:<18}{K:5d}  {name:<14}{100*gk:9.1f}%{100*lc:10.0f}%   {100*rg:.1f}% / {100*rl:.0f}%')
