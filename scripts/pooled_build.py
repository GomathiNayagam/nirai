# Pool the first-8000-read POD5s of the four KPC2 flow cells into ONE stream (round-robin interleave, each flow cell's own order kept) -> pooled.pod5 + order.tsv
import pod5, sys
B = '/home/sgomathi/rawsamble_work/'; out = B + 'kpc2_pool/'
src = {'FAX76295': B + 'kpc2_run/stream/', 'FAX39617': B + 'kpc2_fc/FAX39617/stream/', 'FAX39616': B + 'kpc2_fc/FAX39616/stream/', 'FAV69578': B + 'kpc2_fc/FAV69578/stream/'}
lists = {}
for fc, d in src.items():
    ids = [l.strip() for l in open(d + 'ids_first8000.txt')][:8000]
    with pod5.Reader(d + 'first8000.pod5') as r:
        lists[fc] = {str(rec.read_id): rec.to_read() for rec in r.reads()}
    lists[fc] = [(i, lists[fc][i]) for i in ids if i in lists[fc]]; print(fc, len(lists[fc]), flush=True)
n = max(len(v) for v in lists.values()); rows = []
with pod5.Writer(out + 'pooled.pod5') as w:
    buf = []
    for k in range(n):
        for fc, v in lists.items():
            if k < len(v): buf.append((fc, v[k][0], v[k][1]))
        if len(buf) >= 4000: w.add_reads([b[2] for b in buf]); rows += [(b[0], b[1], len(b[2].signal)) for b in buf]; buf = []
    if buf: w.add_reads([b[2] for b in buf]); rows += [(b[0], b[1], len(b[2].signal)) for b in buf]
with open(out + 'order.tsv', 'w') as o:
    o.write('read_id\tstart_sample\tn_samples\tflowcell\n')
    for i, (fc, rid, ns) in enumerate(rows): o.write(f'{rid}\t{i}\t{ns}\t{fc}\n')
print('pooled', len(rows))
