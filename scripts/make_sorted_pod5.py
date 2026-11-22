"""Merge the batch POD5 files of a run into ONE POD5 whose reads are in acquisition order (sorted by start_sample).
usage: make_sorted_pod5.py <out.pod5> <order.tsv> <in1.pod5> [in2.pod5 ...]"""
import sys, glob, pod5
out, order_tsv, ins = sys.argv[1], sys.argv[2], sys.argv[3:]
reads = []
for f in ins:
    with pod5.Reader(f) as r:
        for rec in r.reads():
            reads.append((rec.start_sample, rec.to_read()))
    print(f.split('/')[-1], len(reads), flush=True)
reads.sort(key=lambda x: x[0])
with pod5.Writer(out) as w, open(order_tsv, "w") as o:
    o.write("read_id\tstart_sample\tn_samples\n")
    for i in range(0, len(reads), 4000):
        w.add_reads([r for _, r in reads[i:i + 4000]])
    for s, r in reads:
        o.write(f"{r.read_id}\t{s}\t{len(r.signal)}\n")
print("done", len(reads))
