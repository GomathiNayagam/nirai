# Evaluate prefix-only (first K samples) overlap against an index of earlier full reads, vs truth coverage from reference mapping.
# usage: prefix_eval.py M N K1,K2,...   (files in kpc2_run/prefix; truth = run/truth_sig.paf)
import sys, numpy as np, collections
M=int(sys.argv[1]); N=int(sys.argv[2]); Ks=[int(x) for x in sys.argv[3].split(',')]
import os; base=os.environ.get('KROOT','/home/sgomathi/rawsamble_work/kpc2_run').rstrip('/')+'/'
order=[l.rstrip('\n').split('\t') for l in open(base+'stream/order.tsv')][1:]
ids=[o[0] for o in order][:N]; ns={o[0]:int(o[2]) for o in order[:N]}
truth={}
for l in open(base+'run/truth_sig.paf'):
    f=l.rstrip('\n').split('\t')
    if f[5]=='*': continue
    truth[f[0]]=(f[4],int(f[7]),int(f[8]))
hist=[r for r in ids[:M] if r in truth]; qry=ids[M:N]
byst={'+':[], '-':[]}
for r in hist: s,a,b=truth[r]; byst[s].append((a,b))
for s in byst: byst[s]=np.array(byst[s])
def depth(s,a,b,W):
    h=byst[s]; ov=np.minimum(h[:,1],b)-np.maximum(h[:,0],a)
    return int((ov>=0.5*W).sum())
def truthhit(s,a,b,t):
    st,ta,tb=truth[t]; return st==s and min(tb,b)-max(ta,a)>0
print(f'history {M} reads ({len(hist)} with truth position); query {len(qry)} reads, {sum(r in truth for r in qry)} with truth position')
for K in Ks:
    W=K*0.08
    hits=collections.defaultdict(set)
    for l in open(base+f'prefix/q_{K}'+__import__('os').environ.get('SUF','')+'.paf'):
        f=l.split('\t')
        if f[5] not in('*',) : hits[f[0]].add(f[5])
    rows=[]
    for r in qry:
        if r not in truth: continue
        s,a,b=truth[r]
        pa,pb=(a,a+W) if s=='+' else (b-W,b)
        d=depth(s,pa,pb,W); H=hits.get(r,set())
        ok=sum(truthhit(s,pa,pb,t) for t in H if t in truth)
        rows.append((r,d,len(H),ok,ns[r]))
    d=np.array([x[1] for x in rows]); h=np.array([x[2] for x in rows]); ok=np.array([x[3] for x in rows]); n=np.array([x[4] for x in rows])
    print(f'\n=== prefix {K} samples = {K/5000:.1f} s = ~{W:.0f} bases; truth depth = same-strand earlier reads covering the prefix region ===')
    print(f'query reads {len(d)}: truth depth 0: {np.mean(d==0)*100:.0f}%  1-2: {np.mean((d>=1)&(d<=2))*100:.0f}%  >=3: {np.mean(d>=3)*100:.0f}%')
    print(f'reads with >=1 hit: {np.mean(h>=1)*100:.0f}%; hit precision (hits that truly overlap): {ok.sum()/max(h.sum(),1)*100:.0f}%')
    print(f'{"reject if hits>=":>16} {"rejected":>9} | {"catch d>=3":>10} {"wrong d=0":>9} {"wrong d=1-2":>11} | {"seq time saved":>14}')
    for t in (1,2,3):
        rej=h>=t
        saved=((n[rej]-K).clip(0)).sum()/n.sum()*100
        print(f'{t:>16} {rej.mean()*100:8.0f}% | {rej[d>=3].mean()*100:9.0f}% {rej[d==0].mean()*100:8.0f}% {rej[(d>=1)&(d<=2)].mean()*100:10.0f}% | {saved:13.0f}%')
