"""Variant-calling accuracy of a read subset against the NanoVarBench KPC2 truth set (reads come from the isolate; aligned to mutreference.fna; truth.vcf = variants that restore the isolate).
Caller = bcftools (pysam 0.23 bundled), haploid.  score(fastq) -> dict(SNP/indel precision, recall, F1, mean depth)."""
import pysam, pysam.bcftools, pysam.samtools, gzip, subprocess, os, tempfile, numpy as np
T = "/home/sgomathi/rawsamble_work/kpc2_truth/KPC2__202310/"; MREF = T + "mutreference.fna"
def norm_vcf_to_set(path, filt=None):
    s = {}
    for rec in pysam.VariantFile(path):
        if filt and not filt(rec): continue
        for alt in rec.alts or []:
            if alt == "<*>" or alt == "*": continue
            s[(rec.chrom, rec.pos, rec.ref, alt)] = rec
    return s
_truth = None
def truth():
    global _truth
    if _truth is None:
        tmp = tempfile.mkdtemp(); out = tmp + "/truth.norm.vcf"
        pysam.bcftools.norm("-f", MREF, "-m", "-any", "-o", out, T + "truth.vcf.gz", catch_stdout=False)
        _truth = norm_vcf_to_set(out)
    return _truth
def call(fastq, outdir, threads=8, minq=20, minaf=0.5, mindp=3):
    os.makedirs(outdir, exist_ok=True); bam = outdir + "/a.bam"
    sam = subprocess.run(["minimap2", "-ax", "map-ont", "-t", str(threads), MREF, fastq], capture_output=True)
    open(outdir + "/a.sam", "wb").write(sam.stdout)
    pysam.sort("-@", str(threads), "-o", bam, outdir + "/a.sam"); pysam.index(bam); os.remove(outdir + "/a.sam")
    raw = outdir + "/raw.vcf"
    pysam.bcftools.mpileup("-f", MREF, "-a", "AD,DP", "-Q", "7", "-q", "5", "-B", "-d", "1000", "-Ou", "-o", outdir + "/pu.bcf", bam, catch_stdout=False)
    pysam.bcftools.call("-mv", "--ploidy", "1", "-Ov", "-o", raw, outdir + "/pu.bcf", catch_stdout=False)
    nrm = outdir + "/norm.vcf"; pysam.bcftools.norm("-f", MREF, "-m", "-any", "-o", nrm, raw, catch_stdout=False)
    def ok(rec):
        dp = rec.info.get("DP", 0); ad = rec.samples[0].get("AD"); 
        if ad is None or dp < mindp or rec.qual is None or rec.qual < minq: return False
        i = 1  # alt count / total depth (multiallelic was split, AD keeps all alleles; use max alt)
        return max(ad[1:]) / max(1, sum(ad)) >= minaf
    calls = norm_vcf_to_set(nrm, ok)
    depth = pysam.samtools.depth("-a", bam, catch_stdout=True)
    d = np.array([int(l.rsplit("\t", 1)[1]) for l in depth.splitlines()]) if depth else np.array([0])
    return calls, float(d.mean())
def score(fastq, outdir, **kw):
    t = truth(); calls, md = call(fastq, outdir, **kw); res = {"mean_depth": round(md, 1)}
    for kind, f in (("SNP", lambda k: len(k[2]) == 1 and len(k[3]) == 1), ("indel", lambda k: len(k[2]) != len(k[3]))):
        tk = {k for k in t if f(k)}; ck = {k for k in calls if f(k)}; tp = len(tk & ck)
        p = tp / max(1, len(ck)); r = tp / max(1, len(tk)); res[kind] = dict(P=round(100 * p, 2), R=round(100 * r, 2), F1=round(200 * p * r / max(1e-9, p + r), 2), calls=len(ck))
    return res
