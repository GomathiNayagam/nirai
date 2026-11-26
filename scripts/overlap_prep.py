"""Signal-position labels for a learned overlap finder (melb2 first 50,000 reads).
Dorado fast v5.2.0 (CPU, --emit-moves) calls each read; the call is aligned to the reference with minimap2; the move table gives the signal sample at which every called base starts.
Result: for each read, for every called base, its sample index and its reference position, so any reference position can be turned into a position in the raw signal of any read covering it.
Output: melb2/moves/labels.pkl  {read_id: dict(chrom, strand, n_samples, base_sample[int32], base_ref[int32; -1 = unaligned])}.   usage: overlap_prep.py"""
import os, sys, pickle, subprocess, numpy as np, pysam
W = os.path.expanduser('~/rawsamble_work/melb2/'); M = W + 'moves/'; REF = W + 'truth/ATCC_17802__202309/reference.fna'
fq = M + 's50k.fastq'; info = {}
if not os.path.exists(fq):
    with pysam.AlignmentFile(M + 's50k.bam', 'rb', check_sq=False) as bam, open(fq, 'w') as o:
        for r in bam.fetch(until_eof=True):
            if r.is_secondary or r.is_supplementary: continue
            mv = r.get_tag('mv'); stride = mv[0]; ts = r.get_tag('ts') if r.has_tag('ts') else 0; mv = np.array(mv[1:], np.int8); starts = ts + stride * np.flatnonzero(mv)
            if len(starts) != len(r.query_sequence): continue
            info[r.query_name] = (r.get_tag('ns') if r.has_tag('ns') else 0, starts.astype(np.int32))
            o.write(f'@{r.query_name}\n{r.query_sequence}\n+\n{"".join(chr(q + 33) for q in r.query_qualities)}\n')
    pickle.dump(info, open(M + 'starts.pkl', 'wb'))
else: info = pickle.load(open(M + 'starts.pkl', 'rb'))
print(len(info), 'reads with a consistent move table', flush=True)
sam = M + 's50k.sam'
if not os.path.exists(sam): subprocess.run(f'minimap2 -a -x map-ont -t 8 --secondary=no {REF} {fq} > {sam} 2> {M}mm2.err', shell=True, check=True)
out = {}; n_all = n_ok = 0
with pysam.AlignmentFile(sam, 'r') as s:
    for r in s.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary or r.query_name not in info: continue
        n_all += 1; ns, starts = info[r.query_name]; L = len(starts)
        if r.mapping_quality < 20 or r.query_alignment_length < 0.5 * L: continue
        ref = np.full(L, -1, np.int32)
        for q, t in r.get_aligned_pairs(matches_only=True):
            ref[(L - 1 - q) if r.is_reverse else q] = t
        out[r.query_name] = dict(chrom=r.reference_name, strand='-' if r.is_reverse else '+', n_samples=ns, base_sample=starts, base_ref=ref); n_ok += 1
pickle.dump(out, open(M + 'labels.pkl', 'wb')); print(f'aligned reads {n_all}; kept (mapq>=20, >=50% aligned) {n_ok}')
# sanity: samples per base and how much of the signal the aligned bases cover
sp = np.array([(v['base_sample'][-1] - v['base_sample'][0]) / max(1, len(v['base_sample'])) for v in out.values()]); lead = np.array([v['base_sample'][0] for v in out.values()])
print(f'samples per base: median {np.median(sp):.1f}, 5-95% {np.percentile(sp, 5):.1f}-{np.percentile(sp, 95):.1f}; samples before the first base: median {np.median(lead):.0f}')
