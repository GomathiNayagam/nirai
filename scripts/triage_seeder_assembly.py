"""Assembly check for triage selections (melb2 first 50,000 reads, D=5). Same read sets the payoff test scored (melb2/gate_eval/seeder/<arm>/reads.fastq, public hac calls).
Flye --nano-hq, genome size 5.2m, identical settings per arm; assembly aligned to the reference with minimap2 -x asm5: contigs, N50, total length, reference covered (%), identity (matches / alignment length over primary alignments).
usage: triage_seeder_assembly.py -> results/nirai_triage_seeder_assembly_melb2.txt"""
import os, subprocess, numpy as np
W = os.path.expanduser('~/rawsamble_work/melb2/'); G = W + 'gate_eval/'; REF = W + 'truth/ATCC_17802__202309/reference.fna'; A = W + 'gate_eval/seeder_asm/'; os.makedirs(A, exist_ok=True)
OUT = open(os.path.expanduser('~/thusumbu_repo/results/nirai_triage_seeder_assembly_melb2.txt'), 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
arms = [('all reads', G + 'seeder/all'), ('Rawsamble rule D=5', G + 'seeder/POLICY5'), ('v2 seeder D=5 (K3 V3, shorts always basecalled)', G + 'seeder/SEEDERk3-long5'), ('  first-come, same bases', G + 'seeder/SEEDERk3-long5_fc'),
        ('  random, same bases', G + 'seeder/SEEDERk3-long5_r0'), ('oracle D=5', G + 'seeder/ORACLE-long5'), ('v2 seeder + crowd filter D=5 (K3 V4)', G + 'seeder/SEEDERc-K3V4-long5'), ('v2 seeder + crowd filter D=5 (K1 V3)', G + 'seeder/SEEDERc-K1V3-long5')]
clen = sum(int(l.split()[1]) for l in open(REF + '.fai')); P(f'\n=== melb2 assembly check, Flye nano-hq, reference {clen/1e6:.2f} Mb ===')
for lab, d in arms:
    fq = d + '/reads.fastq'
    if not os.path.exists(fq): P(f'{lab}: missing {fq}'); continue
    tag = d.rstrip('/').split('/')[-1]; od = A + tag; mb = sum(len(l) for i, l in enumerate(open(fq)) if i % 4 == 1) / 1e6
    if not os.path.exists(od + '/assembly.fasta'): subprocess.run(f'flye --nano-hq {fq} --genome-size 5.2m --threads 12 --out-dir {od} > {A}{tag}.log 2>&1', shell=True)
    if not os.path.exists(od + '/assembly.fasta'): P(f'{lab}: Flye failed (see {A}{tag}.log)'); continue
    L = []; name = None
    for l in open(od + '/assembly.fasta'):
        if l.startswith('>'): L.append(0)
        else: L[-1] += len(l.strip())
    L = sorted(L, reverse=True); tot = sum(L); n50 = next(x for x, c in zip(L, np.cumsum(L)) if c >= tot / 2)
    paf = subprocess.run(f'minimap2 -x asm5 -t 8 {REF} {od}/assembly.fasta', shell=True, capture_output=True, text=True).stdout.splitlines()
    iv = {}; m = b = 0
    for l in paf:
        c = l.split('\t'); 
        if int(c[11]) < 20: continue
        iv.setdefault(c[5], []).append((int(c[7]), int(c[8]))); m += int(c[9]); b += int(c[10])
    cov = 0
    for ch, v in iv.items():
        v.sort(); s, e = v[0]
        for x, y in v[1:]:
            if x > e: cov += e - s; s, e = x, y
            else: e = max(e, y)
        cov += e - s
    P(f'{lab:<52}{mb:6.1f} Mb reads | contigs {len(L):3d}  total {tot/1e6:5.2f} Mb  N50 {n50/1e6:5.2f} Mb | reference covered {100*cov/clen:5.1f}%  identity {100*m/max(1,b):6.3f}%')
P('ASSEMBLY_DONE')
