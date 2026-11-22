# Nirai

**Nirai** (Tamil: fullness, sufficiency; developed within the thusumbu project). Pre-basecalling triage for nanopore sequencing: use only the raw signal (all-vs-all raw-signal overlaps, no basecalls, no reference) to decide which reads to basecall first, and when a region has enough reads, so that basecalling compute, energy and storage are spent where they add information. Aimed at low-resource settings (clinic laptops, space missions).

Scope: ONT R10.4.1 chemistry (5 kHz, 400 bps) and newer. Status: research prototype, not a released tool.

## Idea

1. Overlap each incoming read (POD5) against earlier reads with Rawsamble / RawHash2 in online mode (growing index).
2. From the overlaps, estimate per-window depth of the new read using earlier kept reads.
3. Basecall the read only if its depth deficit against a target D is at least tau (default D = 3-10, tau = 0.25, 1 kb windows). Causal: each decision uses only earlier reads.

## Evidence so far (K. pneumoniae KPC2, four R10.4.1 flow cells, one isolate)

All numbers are from `results/`. Variant calling uses minimap2 + bcftools (a general-purpose caller, not nanopore-trained) against the NanoVarBench truth set.

- Variant calling, first ~8,000 reads per flow cell (8-14x): triage at D=3 or D=5 beats first-come and random subsets of the same number of bases on SNP F1 in all four flow cells (`*_variant_downstream.txt`).
- Pooled 31,222 reads (44.6x, `kpc2_pooled_depth_stopping_test.txt`): SNP F1 plateaus by about 15x; indel F1 keeps rising to about 30x. Triage at D=10 keeps 70% of the bases with indel F1 92.1 against 92.4 for all reads. The margin over first-come is small at this depth.
- The overlap-based depth estimate is about 0.7x the true same-strand depth and is weak for reads under 3 kb.
- Prefix-only (first 0.5-2 s) reference-free rejection is only weakly feasible with current RawHash2 settings (`*_prefix_rejection_replay.txt`).
- Signal-only assembly graphs (miniasm on Rawsamble overlaps) cover less of the genome than basecalled same-strand overlaps, which cover less than both strands (`*_signal_assembly_test.txt`).
- A signal-noise quality gate showed no downstream benefit (`triage_quality_r10.txt`, variant files).

## Install and use

```
pip install -e .            # needs numpy pandas scikit-learn pod5 joblib
nirai train-gate --pass-dir PASS_POD5_DIR --fail-dir FAIL_POD5_DIR --prefix 0.4s -o gate.joblib
nirai gate --model gate.joblib reads/*.pod5 -o decisions.tsv        # basecall or skip each read, from the first 0.4 s of signal
nirai triage --paf online.paf -D 5 --gate decisions.tsv -o chosen.tsv   # redundancy rule on Rawsamble/RawHash2 --online overlaps
```

- **Gate:** classifies a read as likely pass or fail from signal features (noise level, signal spread, outlier fraction) of its first 0.4 to 4 s (no basecalls, no reference). Train it on pass/fail folders of a run with known labels. The saved threshold keeps 95% of held-out pass reads.
- **Triage:** causal redundancy rule (see Idea). Input is the PAF from `rawhash2 -x ava-sensitive --r10 --online`, reads in arrival order. A gate file makes reads marked `skip` ineligible.
- The gate and triage are software steps on recorded POD5 files. Ejecting a read while it is in the pore needs the sequencer's Read Until interface and is not implemented.
- The package reproduces the replay rule used in `scripts/`; `tests/` has unit tests (`pytest`).

## To do

### Top priority: find likely-fail reads in real time

**What and why.** Some reads are of low quality and end up in the "fail" folder. Today the sequencer decides pass or fail only after the read is basecalled, so the basecalling work is already spent and then thrown away. In the RDH275 run, fail reads were about 36% of all bases. If we can tell from the raw signal alone that a read is likely to fail, we can skip basecalling it. This needs no reference and no basecalls, and it works for any sample. It saves compute, energy and storage, which matters most on small machines in clinics and in space.

**What we know so far.**
- The signal of the first 0.4 seconds (about 160 bases) already shows if a read is likely to fail. It uses three simple numbers: how noisy the signal is, how spread out it is, and how many outlier samples it has.
- On the training run (RDH275, files held out), the gate rejects about 80% of fail reads and keeps 95% of pass reads (AUC 0.946).
- It is fast: about 330 reads per second on one CPU process, which is enough for a MinION. A PromethION would need several processes.
- On other runs it rarely throws away good reads (95% to 99% of pass reads kept on KPC2, RDH275 other chunks, and the ladder). It catches fewer bad reads, though: only about 50% of fail reads on the ladder.
- Median current is not used because it depends on the sample (its base composition) and broke the threshold on a new run.

**What is not done.**
- The model is trained on one run only. This is why the catch rate drops from about 80% to about 50% on a new run.
- It works on recorded POD5 files, after each read has ended. It does not yet run live while the read is in the pore.
- We have not measured how much basecalling it saves on a whole run, or whether skipped reads would have mattered for variant calls or assembly.

**Next steps, in order.**
1. Train on several runs (different organisms, flow cells and dates) that have pass and fail folders. Test each time on a run that was not used for training. Report the share of fail reads caught and the share of pass reads kept.
2. Pick a threshold that keeps at least 95% of pass reads on a new run, and check that it stays stable.
3. Split "fail" into its causes (low quality, blocked pore, adapter only) and see which ones the gate catches.
4. Measure the saving on a full run: basecalling time and bases skipped, and check variant calls and assembly against basecalling everything.
5. Add a directory-watching mode so the gate runs on POD5 files as the sequencer writes them.
6. Add Read Until support so a failing read can be ejected from the pore after about 0.4 s (this also saves sequencing time; it needs a sequencer).
7. Check the pass/fail quality threshold that MinKNOW uses for the kit in use, so our labels match.
8. Re-test median current (pA) as a feature. It is left out now because it moved the threshold on new runs (KPC2: 85% of pass reads kept instead of 95%; ladder: 8%), and it added only about 2 points of fail catch rate within the training run. Test two versions with leave-one-run-out on several runs: the absolute value, and a run-relative value (median pA minus the running median of earlier reads in the same run, which needs no labels). Add it back as an option only if it helps on most held-out runs.

### Gate details
- [ ] Train on several runs and chemistry versions, and report held-out-run accuracy (RDH275 alone: AUC 0.95 at 0.4 s with files held out; trained on RDH275 and tested elsewhere: ladder AUC 0.85, see `results/nirai_gate_*`).
- [x] First cross-run test (train RDH275, test ladder pass/fail and KPC2 pass-only). Median pA does not transfer, so the gate uses sd, MAD of differences and outlier fraction. Result: KPC2 95% of pass kept, ladder 99% pass kept but only 50% of fail rejected (AUC 0.85).
- [ ] Test on KPC2 or other runs that have a fail folder, and on a different flow cell type.
- [ ] Separate causes of fail (low quality, blocked pore, adapter-only) and report accuracy per cause.
- [ ] Measure how much basecalling the gate saves on a whole run, and whether it removes any reads that matter downstream (variant calls, assembly).
- [ ] Check the pass/fail threshold definition used by MinKNOW for the kit version in use.
- [ ] Apply the proper offset in the picoampere conversion and retrain (the experiments used scale only).

Redundancy triage
- [ ] Improve overlap sensitivity for reads under 3 kb and noisy reads (largest measured gap to the oracle: Nirai captures about a quarter of the oracle saving on KPC2, and is worse than first-come on RDH275).
- [ ] Find out why RDH275 is worse than first-come (short or unmapped reads selected?); test the gate in front of the triage.
- [ ] Calibrate the depth estimate (about 0.7x the true same-strand depth on KPC2) and turn it into a stopping rule.
- [ ] Index growth and a rule for when to stop adding reads to the index (open problem named in the Rawsamble paper).
- [ ] Ladder test with a reference-free evaluation (basecalls of all reads as scoring truth; Nirai never sees them).
- [ ] RDH275 variant-calling test with frozen parameters.
- [ ] Reverse-strand overlaps (measured gain is small: same-strand and both-strand oracles differ by 0.6 to 1.8 points).

System
- [ ] Measure wall-clock time and energy on low-resource hardware (no measurements yet).
- [ ] Directory-watching interface for live POD5 output.
- [ ] Read Until integration for mid-read rejection.
- [ ] Pooled and multi-isolate benchmark and a second variant caller (only bcftools so far).
- [ ] Add a LICENSE.

## Limits

- Rawsamble overlaps same-strand reads only, so overlap depth is about half of total depth.
- One isolate, one caller. No time or energy measurement on low-resource hardware yet.
- Reads under about 1 kb are almost never overlapped.

## Layout

- `nirai/` the package (gate, triage, CLI); `tests/`
- `scripts/` experiment code (paths are absolute and point to the author's machine; set `KROOT` and edit `P`/`REF` constants to rerun).
- `results/` plain-text outputs quoted above.

## Dependencies

RawHash2 / Rawsamble (https://github.com/CMU-SAFARI/RawHash, GPLv3; the online-mode build used here lives on a private fork and is not included), Dorado, minimap2, miniasm, Flye, pysam (bundled samtools/bcftools), numpy, pandas, scikit-learn.

## Related work

Rawsamble (Bioinformatics 2026), HARICOT (bioRxiv 2026, offline raw-signal assembly), TargetCall (pre-basecalling filtering against a target reference), BOSS-RUNS/AEONS (adaptive sampling with basecalled reads).
