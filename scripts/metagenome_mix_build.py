"""In-silico mock metagenome from three real R10.4.1 isolates with different abundances (real signal, random arrival order, each isolate's own read order kept):
 A = K. pneumoniae KPC2 (pooled stream), B = RDH275 stream, C = E. coli ATCC 25922 (r10_60k).  Source of each read is known, so abundance and per-organism coverage are exact.
usage: metagenome_mix_build.py NA NB NC [seed] [outdir name]  -> meta_mix/mix.pod5, meta_mix/order.tsv"""
import sys, numpy as np, pod5
B = '/home/sgomathi/rawsamble_work/'; out = B + (sys.argv[5] if len(sys.argv) > 5 else 'meta_mix') + '/'
NA, NB, NC = (int(x) for x in sys.argv[1:4]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
def first_n(path, order_ids, n):
    ids = order_ids[:n]; want = set(ids)
    with pod5.Reader(path) as r: d = {str(rec.read_id): rec.to_read() for rec in r.reads(selection=list(want), missing_ok=True)}
    return [(i, d[i]) for i in ids if i in d]
import pandas as pd
oa = pd.read_csv(B + 'kpc2_pool/order.tsv', sep='\t').read_id.tolist()
ob = pd.read_csv(B + 'rdh275/stream/order.tsv', sep='\t').read_id.tolist()
oc = [l.strip() for l in open(B + 'r10_run/ids_60k.txt')]
src = {'KPC2': first_n(B + 'kpc2_pool/pooled.pod5', oa, NA), 'RDH275': first_n(B + 'rdh275/stream/sorted.pod5', ob, NB), 'ECOLI': first_n(B + 'r10_run/r10_60k.pod5', oc, NC)}
print({k: len(v) for k, v in src.items()}, flush=True)
rng = np.random.default_rng(seed); lab = np.concatenate([[k] * len(v) for k, v in src.items()]); rng.shuffle(lab)   # random interleave, within-source order kept
ptr = {k: 0 for k in src}; rows = []; buf = []
with pod5.Writer(out + 'mix.pod5') as w:
    for k in lab:
        rid, rd = src[k][ptr[k]]; ptr[k] += 1; buf.append(rd); rows.append((rid, k, len(rd.signal)))
        if len(buf) >= 4000: w.add_reads(buf); buf = []
    if buf: w.add_reads(buf)
with open(out + 'order.tsv', 'w') as o:
    o.write('read_id\tsource\tn_samples\n')
    for r in rows: o.write('\t'.join(map(str, r)) + '\n')
print('mix', len(rows))
