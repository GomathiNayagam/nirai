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
- **PromethION / ligation kit (HG002, FLO-PRO114M, SQK-LSK114): does not transfer.** AUC 0.57-0.62 (labels from the MinKNOW pass/fail folder, 6 pod5 files, so this test is weak). HG002 is not in the training data. Do not use the gate on this hardware or kit.
- Not tested on R9.4, rapid-barcode runs (the earlier 0.4 s gate failed there), GridION, or eukaryotes other than HG002.
- Needs about 500 reads of a run before it can score (run-relative features), and reads shorter than 1.0 s are not scored.
- Labels are mean Q from basecalls of different Dorado versions; "Q < 10" is not exactly the same thing across runs.
