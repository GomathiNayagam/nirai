"""Learned decision on top of RawHash2/Rawsamble overlaps (melb2 first 50,000 reads).  The overlap finder is unchanged; a small boosted-tree model replaces the hand-made depth rule for reads >= 1 kb.
Inputs per read (causal, only earlier reads): length, number of overlap records, depth from CHOSEN earlier reads (mean, share of windows with none, share below D), depth from ALL earlier reads, overlap lengths,
chain scores / minimizer counts / mapq of the overlaps, how much of the run has been chosen so far.   Label = oracle decision (true reference positions + earlier chosen reads, same rule).
Time split: model trained on reads 0-25k, then run closed-loop (its own decisions feed its next features) on reads 25k-50k.   usage: triage_learned.py [D] [--variants]
Writes results/nirai_triage_learned_melb2.txt"""
import os, sys, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/scripts', 1)[0])
from nirai.triage import window_depth, length_recall, EVENTS_PER_BASE
W = os.path.expanduser('~/rawsamble_work/melb2/'); D = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 5; TAU = 0.25; WIN = 1000; BIN = 100; SPLIT = 25000
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_triage_learned_melb2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
# ---- full PAF records
order, qlen, recs = {}, {}, {}
for l in open(W + 'fc50k/run/online.paf'):
    c = l.rstrip('\n').split('\t'); q = c[0]
    if q not in order: order[q] = len(order); qlen[q] = int(c[1]); recs[q] = []
    if c[5] != '*':
        t = {x.split(':')[0]: float(x.split(':')[2]) for x in c[12:]}; recs[q].append((c[5], int(c[2]), int(c[3]), int(c[10]), int(c[11]), t.get('cm', 0), t.get('s1', 0)))
names = sorted(order, key=order.get); N = len(names); idx = {r: i for i, r in enumerate(names)}; nb = np.array([qlen[r] / EVENTS_PER_BASE for r in names])
best = {}
for l in open(W + 'fc50k/run/probe_truth.paf'):
    c = l.split('\t'); r = c[0]; m = int(c[9])
    if r not in best or m > best[r][0]: best[r] = (m, c[4], c[5], int(c[7]), int(c[8]), int(c[1]), int(c[3]) - int(c[2]))
clen = {l.split()[0]: int(l.split()[1]) for l in open(W + 'truth/ATCC_17802__202309/reference.fna.fai')}
has_t = np.array([r in best and best[r][6] >= 0.5 * best[r][5] for r in names]); ones = np.ones(N, bool)

def feats(i, chosen, n_ch):
    r = names[i]; rc = recs[r]; q = qlen[r]; e = window_depth(q, [(t, a, b) for t, a, b, *_ in rc], chosen, idx, WIN); ea = window_depth(q, [(t, a, b) for t, a, b, *_ in rc], ones, idx, WIN)
    ch = [x for x in rc if chosen[idx[x[0]]]] if rc else []; ln = np.array([(b - a) / q for _, a, b, *_ in ch]) if ch else np.zeros(1)
    cm = np.array([x[5] for x in ch]) if ch else np.zeros(1); s1 = np.array([x[6] for x in ch]) if ch else np.zeros(1); mq = np.array([x[4] for x in ch]) if ch else np.zeros(1)
    ec = np.minimum(e / length_recall(nb[i]), D) if nb[i] >= 1000 else e
    return [nb[i], len(rc), len(ch), e.mean(), (e == 0).mean(), (e < D).mean(), ea.mean(), (ea == 0).mean(), ec.mean(), np.maximum(0, D - ec).mean() / D, ln.sum(), ln.max(), cm.sum(), cm.max(), s1.sum(), s1.max(), mq.mean(), (mq >= 60).mean(), n_ch / (i + 1), i / N]

def run(mode, model=None, thr=0.5, label_cov=True):
    """one causal pass; mode 'policy' = nirai rule; 'model' = learned (reads >= SPLIT use the model, earlier reads use the rule).  Returns chosen, oracle label, features."""
    cov = {(c, s): np.zeros(clen[c] // BIN + 2, np.int32) for c in clen for s in '+-'}; chosen = np.zeros(N, bool); orc = np.zeros(N, bool); X = np.zeros((N, 20)); pr = np.full(N, np.nan); n_ch = 0
    for i, r in enumerate(names):
        X[i] = feats(i, chosen, n_ch)
        if nb[i] < 1000: chosen[i] = True
        elif mode == 'model' and i >= SPLIT: pr[i] = model.predict_proba(X[i:i + 1])[0, 1]; chosen[i] = pr[i] >= thr
        else: chosen[i] = X[i][9] >= TAU
        if has_t[i]:
            _, st, ch, ts, te, *_ = best[r]; nW = max(1, -(-qlen[r] // WIN)); k = np.clip((np.arange(nW) + .5) * WIN / qlen[r], 0, 1)
            pos = ts + k * (te - ts) if st == '+' else te - k * (te - ts); c = cov[(ch, st)]; td = c[np.clip((pos // BIN).astype(int), 0, len(c) - 1)]
            orc[i] = True if nb[i] < 1000 else np.maximum(0, D - np.minimum(td, D)).mean() / D >= TAU
            if chosen[i]: c[ts // BIN: te // BIN + 1] += 1
        else: orc[i] = True
        n_ch += chosen[i]
    return chosen, orc, X, pr

P(f'\n=== learned decision on top of Rawsamble, melb2 first {N} reads, D={D}, tau={TAU}; train reads 0-{SPLIT}, closed-loop test {SPLIT}-{N} ===')
ch0, orc0, X0, _ = run('policy'); tr = (np.arange(N) < SPLIT) & (nb >= 1000) & has_t; te = (np.arange(N) >= SPLIT) & (nb >= 1000) & has_t
P(f'policy: chosen {100*ch0.mean():.1f}% of reads; train reads {int(tr.sum())}, test reads {int(te.sum())}; oracle-choose share in test {100*orc0[te].mean():.1f}%')
mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0).fit(X0[tr], orc0[tr])
p = mdl.predict_proba(X0[te])[:, 1]
P(f'open-loop AUC for the oracle decision on test reads (chosen sets from the rule): learned {roc_auc_score(orc0[te], p):.3f} | rule deficit score {roc_auc_score(orc0[te], X0[te][:, 9]):.3f} | estimated depth (inverted) {roc_auc_score(orc0[te], -X0[te][:, 3]):.3f}')
names_f = ['len', 'n_rec', 'n_rec_chosen', 'depth_chosen', 'share0_chosen', 'shareLTD_chosen', 'depth_all', 'share0_all', 'depth_corr', 'deficit', 'ovl_frac_sum', 'ovl_frac_max', 'cm_sum', 'cm_max', 's1_sum', 's1_max', 'mapq_mean', 'mapq60', 'chosen_so_far', 'arrival']
from sklearn.inspection import permutation_importance
pi = permutation_importance(mdl, X0[te], orc0[te], n_repeats=3, random_state=0, scoring='roc_auc'); top = np.argsort(-pi.importances_mean)[:8]
P('what the learned model leans on (drop in AUC when shuffled): ' + ', '.join(f'{names_f[j]} {pi.importances_mean[j]:.3f}' for j in top))
def report(tag, ch, orc):
    m = te; ag = (ch[m] == orc[m]).mean() * 100; w = (ch[m] & ~orc[m]).mean() * 100; lo = (~ch[m] & orc[m]).mean() * 100
    P(f'  {tag:<26} chosen (test reads >=1 kb) {100*ch[m].mean():5.1f}% | agree with oracle {ag:5.1f}% | wasted {w:4.1f}% lost {lo:4.1f}% | chosen bases in 2nd half {100*nb[SPLIT:][ch[SPLIT:]].sum()/nb[SPLIT:].sum():5.1f}%')
report('rule (policy)', ch0, orc0)
runs = {'policy': ch0}
for thr in (0.85, 0.9, 0.95):
    ch, orc, _, _ = run('model', mdl, thr); report(f'learned thr {thr}', ch, orc); runs[f'learned{thr}'] = ch
np.savez(W + 'gate_eval/triage_learned_choices.npz', **runs)
P('LEARNED_DONE')
