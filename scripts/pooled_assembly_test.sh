#!/bin/bash
# Flye assembly of basecalled reads kept by each policy (pooled KPC2, from pooled_depth_test.py selections) vs reference
P=/home/sgomathi/rawsamble_work/kpc2_pool; REF=/home/sgomathi/rawsamble_work/kpc2_truth/KPC2__202310/reference.fna
OUT=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/results/kpc2_pooled_assembly_test.txt
for t in p3_all p3_tri10 p3_tri10_rnd0 p3_tri10_fc p3_tri8 p3_tri8_rnd0 p3_tri5 p3_tri5_rnd0 p3_tri5_fc; do
  d=$P/asm/$t; [ -s $d/assembly.fasta ] || { rm -rf $d; nice -n 10 flye --nano-hq $P/ds/$t/reads.fastq -g 5.4m -t 16 -o $d >$P/asm/$t.log 2>&1; }
  minimap2 -x asm5 -c --cs -t 8 $REF $d/assembly.fasta > $d/vs_ref.paf 2>/dev/null
  python3 - "$t" "$d" "$REF" "$P/ds/$t/reads.fastq" >> $OUT <<'PY'
import sys,re,collections
t,d,ref,fq=sys.argv[1:5]
G=sum(int(l.split('\t')[1]) for l in open(ref+'.fai'))
L=[]; 
for l in open(d+'/assembly.fasta'):
    pass
seq=collections.OrderedDict(); k=None
for l in open(d+'/assembly.fasta'):
    if l[0]=='>': k=l[1:].split()[0]; seq[k]=0
    else: seq[k]+=len(l.strip())
L=sorted(seq.values(),reverse=True); tot=sum(L); c=0; n50=0
for x in L:
    c+=x
    if c>=tot/2: n50=x;break
iv=[]; mm=ins=dele=aln=0; best={}
for l in open(d+'/vs_ref.paf'):
    f=l.split('\t'); 
    if int(f[11])<20: continue
    iv.append((int(f[7]),int(f[8])))
    a=int(f[10]); aln+=a
    cs=[x for x in f if x.startswith('cs:Z:')][0][5:]
    for m in re.finditer(r'\*[a-z][a-z]|\+[a-z]+|-[a-z]+',cs):
        s=m.group(0)
        if s[0]=='*': mm+=1
        elif s[0]=='+': ins+=len(s)-1
        else: dele+=len(s)-1
iv.sort(); cov=0; cs_=ce=None
for s,e in iv:
    if cs_ is None: cs_,ce=s,e
    elif s<=ce: ce=max(ce,e)
    else: cov+=ce-cs_; cs_,ce=s,e
if cs_ is not None: cov+=ce-cs_
nb=sum(1 for i,l in enumerate(open(fq)) if i%4==1)
bases=sum(len(l.strip()) for i,l in enumerate(open(fq)) if i%4==1)
print(f"{t:<16} reads {nb:6d} {bases/1e6:5.0f} Mb ({bases/G:4.1f}x) | contigs {len(L):3d} total {tot/1e6:5.2f} Mb largest {L[0]/1e6:5.2f} Mb NG50 {n50/1e6:5.2f} Mb | ref covered {cov/G*100:5.1f}% | mismatch {mm/max(aln,1)*1e5:6.1f} ins {ins/max(aln,1)*1e5:6.1f} del {dele/max(aln,1)*1e5:6.1f} per 100kb aligned",flush=True)
PY
done
echo ASM_DONE >> $P/asm/status
