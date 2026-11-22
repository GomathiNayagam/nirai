# Build history POD5 (first M reads, full signal) and prefix POD5s (reads M..N, first K samples) from first8000.pod5 in stream order.
import pod5, dataclasses, sys
M=int(sys.argv[1]); N=int(sys.argv[2]); Ks=[int(x) for x in sys.argv[3].split(',')]
import os; base=os.environ.get('KROOT','/home/sgomathi/rawsamble_work/kpc2_run').rstrip('/')+'/'
ids=[l.split()[0] for l in open(base+'stream/ids_first8000.txt')][:N]
order=[l.split('\t')[0] for l in open(base+'stream/order.tsv')][1:]
assert ids==order[:N], 'ids_first8000 not in stream order'
hist=set(ids[:M]); qry=set(ids[M:N])
out=base+'prefix/'
wh=pod5.Writer(out+f'hist_{M}.pod5')
wq={K:pod5.Writer(out+f'q_{M}_{N}_{K}.pod5') for K in Ks}
nh=nq=0
with pod5.Reader(base+'stream/first8000.pod5') as r:
    for rec in r.reads():
        rid=str(rec.read_id)
        if rid in hist:
            wh.add_read(rec.to_read()); nh+=1
        elif rid in qry:
            rd=rec.to_read(); nq+=1
            for K in Ks: wq[K].add_read(dataclasses.replace(rd, signal=rd.signal[:K]))
wh.close(); [w.close() for w in wq.values()]
print('hist',nh,'query',nq)
