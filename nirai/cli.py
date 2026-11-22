import argparse, sys
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nirai", description="Pre-basecalling triage for nanopore signal (R10.4.1+)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-gate", help="train the pass/fail gate from pass and fail POD5 folders")
    t.add_argument("--pass-dir", nargs="+", required=True); t.add_argument("--fail-dir", nargs="+", required=True)
    t.add_argument("--prefix", default="0.4s", choices=["0.4s", "0.8s", "2s", "4s"]); t.add_argument("--pass-recall", type=float, default=0.95)
    t.add_argument("--max-reads", type=int, default=None, help="per POD5 file, for quick tests"); t.add_argument("-o", "--out", required=True)
    g = sub.add_parser("gate", help="score POD5 files: basecall or skip each read")
    g.add_argument("--model", required=True); g.add_argument("pod5", nargs="+"); g.add_argument("-o", "--out", required=True)
    g.add_argument("--threshold", type=float, default=None)
    r = sub.add_parser("triage", help="redundancy triage from an online Rawsamble/RawHash2 PAF (reads in arrival order)")
    r.add_argument("--paf", required=True); r.add_argument("-D", type=int, default=5, help="wanted same-strand depth"); r.add_argument("--tau", type=float, default=0.25)
    r.add_argument("--window", type=int, default=1000); r.add_argument("--gate", help="decisions.tsv from `nirai gate`; reads marked skip are not eligible")
    r.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train-gate":
        from .gate import train, save_model
        m = train(a.pass_dir, a.fail_dir, a.prefix, a.pass_recall, a.max_reads); save_model(m, a.out); c = m["cv"]
        print(f"trained on {c['n_pass']} pass / {c['n_fail']} fail reads from {c['n_files']} files; grouped-CV AUC {c['auc']:.3f}; "
              f"at the saved threshold {c['pass_kept']*100:.0f}% of pass kept, {c['fail_rejected']*100:.0f}% of fail rejected")
    elif a.cmd == "gate":
        from .gate import load_model, score_pod5
        m = load_model(a.model); n = k = 0
        with open(a.out, "w") as o:
            o.write("read_id\tscore\tdecision\n")
            for rid, s, d in score_pod5(m, a.pod5, a.threshold):
                o.write(f"{rid}\t{s:.4f}\t{d}\n"); n += 1; k += d == "skip"
        print(f"{n} reads scored, {k} marked skip ({k/max(n,1)*100:.1f}%)")
    else:
        from .triage import read_paf, triage
        names, qlen, recs = read_paf(a.paf); elig = None
        if a.gate:
            dec = dict(l.rstrip("\n").split("\t")[::2] for l in list(open(a.gate))[1:])
            elig = np.array([dec.get(n, "basecall") != "skip" for n in names])
        ch = triage(names, qlen, recs, a.D, a.tau, a.window, elig)
        with open(a.out, "w") as o:
            o.write("read_id\tdecision\n")
            for n, c in zip(names, ch): o.write(f"{n}\t{'basecall' if c else 'back_burner'}\n")
        print(f"{len(names)} reads, {int(ch.sum())} chosen for basecalling ({ch.mean()*100:.1f}% of reads)")


if __name__ == "__main__":
    sys.exit(main())
