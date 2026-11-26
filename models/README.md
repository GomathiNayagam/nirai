# Nirai 1.0 s CNN pass/fail gate (`nirai_gate_cnn_1.0s.pt`)

A small 1D CNN (34k parameters, 160 KB) that reads the first 1.0 s (5,000 samples at 5 kHz) of a read's raw signal and returns P(pass), where "fail" means mean read quality Q < 10.
Research prototype. It is not a released tool and has not been run live on a sequencer.

## Use
```python
from nirai.gate_cnn import load_cnn_gate
g = load_cnn_gate("models/nirai_gate_cnn_1.0s.pt")
ref = g.run_reference(first_windows)          # first 500 reads of the run (arrival order, >= 5000 samples each); hold them until you have 500
p_pass = g.score(windows, ref, kit="native")  # windows: (n, >=5000) array of pA values; returns P(pass)
```
Use `nirai.gate.CalibratedGate` on `p_pass` for the threshold. A threshold taken from the training data is NOT reliable on a new run (see below).

## Training data
5 runs from earlier work (KPC2 x3 flow cells, E. coli ATCC 25922, RDH275 + extra fail reads), melb2 (V. parahaemolyticus) and five NanoVarBench isolates (C. jejuni, L. ivanovii, S. aureus, M. tuberculosis, Salmonella).
All are R10.4.1 MinION runs with the native barcode kit (SQK-NBD114-96). Labels: Q < 10 from Dorado hac calls (our own, or the public NanoVarBench calls). Reads without a call are left out. Each new set is capped at 30,000 training reads.
The kit input has three slots (native / rapid / ligation), but only **native** was seen in training. The other two are not supported.

## What was measured (`results/nirai_gate_varied_eval.txt`, `results/nirai_gate_learning_curve_melb2.txt`)
Every number below is on a run the scoring model never saw in training. The shipped file is the final model, trained on all of the above, so it has no held-out test of its own. Its held-out estimates are the leave-one-out rows.
- AUC 0.83-0.94 in leave-one-out on melb2 and the five species (melb2 0.89, C. jejuni 0.84, L. ivanovii 0.91, S. aureus 0.86, Salmonella 0.94, M. tuberculosis 0.83). M. tuberculosis is the weakest and gains the most from other species in training (0.76 without them).
- Longer window helps more than more data: 1.0 s beats 0.4 s by about 0.07 AUC on melb2 for the same training data.
- Adding other species to training did not clearly beat the 5-run model on held-out species (differences are within about 0.02, except M. tuberculosis +0.07).
- **A threshold carried over from training is not safe on a new run.** At the strict setting (0.2% of training good reads flagged) the share of good reads flagged on a new run ranged from 0.01% to 2.8% (the high end is M. tuberculosis), and the share of flagged reads that were really low from 36% to 99%. Use a label-free threshold (`CalibratedGate`, `--anchor median`) instead; that combination has not been re-tested with this CNN.
- Low-quality reads are rare in most runs (2-13%), so the saving on basecalling is small.

## Limits
- **PromethION / ligation kit (HG002, FLO-PRO114M, SQK-LSK114): ranks reads, thresholds not tested.** This file was NOT trained on HG002. On 4 HG002 runs (2 flow cells, 64 pod5 files, labels = MinKNOW pass/fail folder, about 50% fail) an equivalent model that never saw HG002 gave AUC 0.80-0.90 (mean of 3 seeds); training on HG002 too gives 0.83-0.93. The first test in an earlier version of this file (AUC 0.57-0.62, 6 small files) was too weak and is withdrawn. What is NOT tested: a fixed or label-free threshold on PromethION, a pass-heavy run, and other kits; in the HG002 sample the start time alone predicts the folder with AUC about 0.9, so treat the numbers as ranking skill only (inside matched time bins the model scores 0.82-0.92). See `results/nirai_gate_hg002_promethion.jsonl`.
- **What it looks at:** mostly fast sample-to-sample noise (jitter) and high-frequency power in the first second, spread over the whole window; summary inputs add little (`results/nirai_gate_probe_hg002_melb2.txt`). It misses bad reads that look normal in the first second.
- Not tested on R9.4, rapid-barcode runs (the earlier 0.4 s gate failed there), GridION, or eukaryotes other than HG002.
- Needs about 500 reads of a run before it can score (run-relative features), and reads shorter than 1.0 s are not scored.
- Labels are mean Q from basecalls of different Dorado versions; "Q < 10" is not exactly the same thing across runs.

## Overlap seeder (research, not packaged)

A separate small model for the triage stage, not the gate. 333k-parameter 1D CNN that maps a 2000-sample signal chunk (about 160 bases) to a 64-number fingerprint; chunks of different reads at the same genome position get similar fingerprints, so overlaps between reads can be found without basecalls or a reference. Trained on melb2 (first 50,000 reads) with Dorado move tables and minimap2 for labels only. On a held-out part of the same genome it finds 55% / 81% / 94% of true overlaps of 300-500 / 500-1,000 / 1,000-2,000 bases at 92.9% precision (Rawsamble: 1% / 11% / 46% at 88.2%). v2 (trained with speed and noise augmentation) keeps the same overlap results (precision 93.0%; recall 55% / 82% / 95% / 98%) and survives added noise (recall@1 77% at noise 0.3, v1 8%). Limits: one genome and one run so far; slow pores (16+ samples per base) and repeat regions are weak; speed and memory not measured; benefit for triage not yet shown. Code: `scripts/overlap_*.py`; results: `results/nirai_overlap_seeder_*.txt`. Weights are not in the repo yet.
