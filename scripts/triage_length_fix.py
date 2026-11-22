"""Length-corrected depth estimate for triage (frozen from KPC2: estimated/true depth by read length 0.02/0.21/0.50/0.63/0.76 for <1/1-2/2-4/4-8/>8 kb, floor 0.2).
Variants (D frozen, tau=0.25, window 1000 events, causal):
  V0 original;  V1 estimated depth divided by the recall for the read's length (capped at D);  V2 reads shorter than 1 kb are never selected;  V3 = V1 + V2.
Scored on reference alignments (never seen by the rule): genome coverage of the selected reads vs first-come and random at the same bases.
usage: triage_length_fix.py <online.paf> <truth_ref.paf> [maxreads]"""
import sys, os, numpy as np
from collections import defaultdict
paf, tref = sys.argv[1:3]; MAXN = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
os.environ['TRI_W'] = '1000'; os.environ['CONTIG'] = os.environ.get('CONTIG', '')
sys.argv = [sys.argv[0], paf, tref, '3', '0.25']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triage_sim.py')).read().split('print(f"stream:')[0]
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triage_sim.py')): raise SystemExit('triage_sim.py not found next to this script')
ns = {'__file__': __file__}; exec(compile(src, 'ts', 'exec'), ns)
names, N, recs, order, qlen, bases, coverage, W = (ns[k] for k in ('names', 'N', 'recs', 'order', 'qlen', 'bases', 'coverage', 'W'))
N = min(N, MAXN); names = names[:N]; tot = bases[:N].sum()
def recall(L): return 0.2 if L < 1000 else 0.21 if L < 2000 else 0.50 if L < 4000 else 0.63 if L < 8000 else 0.76
def depth(r, chosen):
    nW = max(1, -(-qlen[r] // W)); diff = np.zeros(nW + 1, np.int32); by_t = defaultdict(list)
    for t, qs, qe in recs[r]:
        j = order.get(t)
        if j is not None and j < N and chosen[j]: by_t[t].append((qs, qe))
    for t, iv in by_t.items():
        iv.sort(); cs, ce = iv[0]
        for s, e in iv[1:] + [(None, None)]:
            if s is not None and s <= ce: ce = max(ce, e); continue
            lo, hi = cs // W, min(nW, -(-ce // W))
            if hi > lo: diff[lo] += 1; diff[hi] -= 1
            if s is not None: cs, ce = s, e
    return np.cumsum(diff)[:nW].astype(float)
def run(D, corr, skip_short, tau=0.25, short_always=False):
    chosen = np.zeros(N, bool)
    for i, r in enumerate(names):
        L = bases[i]
        if skip_short and 0 < L < 1000: continue
        if short_always and 0 < L < 1000: chosen[i] = True; continue
        e = depth(r, chosen)
        if corr: e = np.minimum(e / recall(L), D)
        if np.maximum(0, D - e).mean() / D >= tau: chosen[i] = True
    return np.where(chosen)[0]
def cov_line(sel):
    c = coverage(sel); return c.mean(), 100 * (c >= 5).mean(), 100 * (c >= 10).mean()
rng = np.random.default_rng(1); cum = np.cumsum(bases[:N])
print(f'stream {N} reads, {tot/1e6:.1f} Mb')
a = coverage(np.arange(N)); print(f'ALL reads: mean depth {a.mean():.1f}, >=5x {100*(a>=5).mean():.1f}%, >=10x {100*(a>=10).mean():.1f}%')
for D in (3, 5, 10):
    print(f'\n== D={D} (tau 0.25, frozen) ==\n{"policy":<30}{"bases":>7}{"mean depth":>11}{">=5x":>8}{">=10x":>8}')
    for name, corr, sk in (('V0 original', 0, 0), ('V1 length-corrected', 1, 0), ('V2 skip reads <1 kb', 0, 1), ('V3 V1 + V2', 1, 1), ('V4 V1 + short reads always basecalled', 1, 2)):
        sel = run(D, corr, sk == 1, short_always=(sk == 2)); b = bases[sel].sum(); m, f5, f10 = cov_line(sel)
        print(f'{name:<30}{100*b/tot:6.1f}%{m:11.1f}{f5:7.1f}%{f10:7.1f}%', flush=True)
        fc = np.arange(int(np.searchsorted(cum, b)) + 1); m, f5, f10 = cov_line(fc); print(f'{"   first-come, same bases":<30}{"":>7}{m:11.1f}{f5:7.1f}%{f10:7.1f}%')
        rr = []
        for k in range(5): p = rng.permutation(N); rr.append(cov_line(p[: int(np.searchsorted(np.cumsum(bases[p]), b)) + 1]))
        m, f5, f10 = np.mean(rr, 0); print(f'{"   random, same bases (5)":<30}{"":>7}{m:11.1f}{f5:7.1f}%{f10:7.1f}%')
