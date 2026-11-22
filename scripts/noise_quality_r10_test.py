import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
R="/home/sgomathi/rawsamble_work/r10_run/"
sf=pd.read_csv(R+"sigfeat_60k.tsv",sep="\t")
rows={}
for l in open(R+"truth_c_60k.paf"):
    c=l.split("\t")
    if c[0] in rows: continue
    rows[c[0]]=(int(c[1]),int(c[9])/int(c[10]),int(c[11]),c[5])
t=pd.DataFrame.from_dict(rows,orient="index",columns=["bases","ident","mapq","ctg"]).rename_axis("read").reset_index()
d=sf.merge(t,on="read",how="left")
print("reads with signal features:",len(d),"| with primary alignment:",d.ident.notna().sum())
d["mapped"]=d.ident.notna()&(d.mapq>=20)
d["noise"]=d.mad_diff; d["nrel"]=d.mad_diff/d.sd
# 1) mapping rate and identity by noise quintile (all reads incl. unaligned)
d["q"]=pd.qcut(d.noise,5,labels=["Q1 quietest","Q2","Q3","Q4","Q5 noisiest"])
g=d.groupby("q",observed=True).agg(n=("read","size"),mapped=("mapped","mean"),median_len=("bases","median"))
a=d[d.mapped].groupby("q",observed=True).apply(lambda x:pd.Series({"mean_ident":np.average(x.ident,weights=x.bases),"err_%":100*(1-np.average(x.ident,weights=x.bases))}))
print("\nBy noise quintile (mad_diff):\n",g.join(a).round(3))
m=d[d.mapped].copy(); print("\nanalysed reads (mapq>=20):",len(m))
print("Spearman noise vs identity: %+.3f   (rel noise mad/sd: %+.3f)   length vs identity: %+.3f   noise vs length: %+.3f"%(
 spearmanr(m.noise,m.ident)[0],spearmanr(m.nrel,m.ident)[0],spearmanr(m.bases,m.ident)[0],spearmanr(m.noise,m.bases)[0]))
m["lb"]=pd.qcut(m.bases,5,duplicates="drop")
print("\nSpearman noise vs identity WITHIN length quintiles (controls length):")
for k,x in m.groupby("lb",observed=True): print("  len %-18s n=%5d  %+.3f"%(k,len(x),spearmanr(x.noise,x.ident)[0]))
# 2) policy test: choose budget fraction of reads; split halves: pick rule weights on first half, evaluate on second
n=len(m); h=n//2; A=m.iloc[:h]; B=m.iloc[h:].reset_index(drop=True)
def score(x,w): return -w*rankdata(x.noise)/len(x)+(1-w)*rankdata(x.bases)/len(x)   # high = quiet & long
rng=np.random.default_rng(0)
def pick(x,pol,frac):
    k=int(len(x)*frac)
    if pol=="random": i=rng.permutation(len(x))[:k]
    elif pol=="longest": i=np.argsort(-x.bases.values)[:k]
    elif pol=="quietest": i=np.argsort(x.noise.values)[:k]
    else: i=np.argsort(-score(x,float(pol.split("=")[1])))[:k]
    return x.iloc[i]
def stats(s,tot):
    bs=s.bases.sum(); e=100*(1-np.average(s.ident,weights=s.bases)); good=s.bases[s.ident>=0.98].sum()
    return bs/tot*100,e,good/bs*100
# choose w on A at 25% budget minimising error per base... report a grid on B instead (no tuning claim)
print("\nHELD-OUT second half (%d reads, %.1f Mb). Budget = fraction of READS basecalled; columns: %% of all bases basecalled / error %% / %% of chosen bases from reads >=98%% identity"%(len(B),B.bases.sum()/1e6))
tot=B.bases.sum(); print("all reads: error %.2f%%"%(100*(1-np.average(B.ident,weights=B.bases))))
for frac in (0.1,0.25,0.5):
    print(" budget %d%% of reads"%(frac*100))
    for pol in ("random","longest","quietest","combo=0.5","combo=0.7"):
        r=[stats(pick(B,pol,frac),tot) for _ in range(20 if pol=="random" else 1)]
        r=np.mean(r,axis=0); print("   %-10s bases %5.1f%%  error %5.2f%%  good-read share %5.1f%%"%((pol,)+tuple(r)))
# bootstrap CI for noise-vs-identity correlation and quietest-25% error advantage on B
bs=[];adv=[]
for _ in range(300):
    x=B.iloc[rng.integers(0,len(B),len(B))]
    bs.append(spearmanr(x.noise,x.ident)[0])
    q=pick(x,"quietest",0.25); adv.append(100*(np.average(x.ident,weights=x.bases)-np.average(q.ident,weights=q.bases))*-1)
print("\nbootstrap 95%% CI: Spearman(noise,identity) on held-out half [%+.3f, %+.3f]; error reduction of quietest 25%% vs all reads [%.2f, %.2f] points"%(
 *np.percentile(bs,[2.5,97.5]),*np.percentile(adv,[2.5,97.5])))
