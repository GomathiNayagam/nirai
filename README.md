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

## Limits

- Rawsamble overlaps same-strand reads only, so overlap depth is about half of total depth.
- One isolate, one caller. No time or energy measurement on low-resource hardware yet.
- Reads under about 1 kb are almost never overlapped.

## Layout

- `scripts/` experiment code (paths are absolute and point to the author's machine; set `KROOT` and edit `P`/`REF` constants to rerun).
- `results/` plain-text outputs quoted above.

## Dependencies

RawHash2 / Rawsamble (https://github.com/CMU-SAFARI/RawHash, GPLv3; the online-mode build used here lives on a private fork and is not included), Dorado, minimap2, miniasm, Flye, pysam (bundled samtools/bcftools), numpy, pandas, scikit-learn.

## Related work

Rawsamble (Bioinformatics 2026), HARICOT (bioRxiv 2026, offline raw-signal assembly), TargetCall (pre-basecalling filtering against a target reference), BOSS-RUNS/AEONS (adaptive sampling with basecalled reads).
