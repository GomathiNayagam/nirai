# reference position of each read from rawhash2 signal->reference mapping (position = mapped start, +/- strand, extended by est. bases n_samples*0.08)
# usage: kpc2_mk_truth.py <run_root>   (needs stream/order.tsv, run/map_ref2.paf; writes run/truth_sig.paf)
import sys, pandas as pd
R = sys.argv[1].rstrip('/') + '/'
o = pd.read_csv(R + "stream/order.tsv", sep="\t").set_index("read_id"); est = (o.n_samples * 0.08).to_dict()
out = open(R + "run/truth_sig.paf", "w"); n = 0
for l in open(R + "run/map_ref2.paf"):
    c = l.split("\t"); r = c[0]; B = int(est[r])
    if c[5] == "*": out.write(f"{r}\t{B}\t0\t0\t+\t*\t0\t0\t0\t0\t0\t0\n"); continue
    qs, qe = int(c[2]), int(c[3]); s, e, rl = int(c[7]), int(c[8]), int(c[6])
    sc = (e - s) / max(1, qe - qs)
    if c[4] == "+": a = s - qs * sc; b = a + B
    else: b = e + qs * sc; a = b - B
    a = int(max(0, a)); b = int(min(rl, b)); n += 1
    out.write(f"{r}\t{B}\t0\t{B}\t{c[4]}\t{c[5]}\t{rl}\t{a}\t{b}\t0\t0\t{c[11]}\n")
print("mapped", n)
