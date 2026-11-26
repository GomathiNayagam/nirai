# Overlap seeder weights (work in progress, backup snapshot, 26 Nov 2026)

Research weights, not a packaged tool. Each file is a 333k-parameter 1D CNN (class `Enc` in `scripts/overlap_train.py`) that maps a 2000-sample signal chunk to a 64-d unit vector. Trained on melb2 only (first 50,000 reads, labels from Dorado fast `--emit-moves` + minimap2).

- `overlap_enc.pt` v1 (3000 steps, no augmentation); `overlap_enc_v2.pt` v2 (6000 steps, stretch/noise augmentation, last 25% of chromosome_1 held out).
- `overlap_enc_cv{0..3}.pt`: v2 settings with quarter q of EVERY chromosome held out (cross-fitting for the triage payoff test, `scripts/triage_seeder_recs.py`).

Status: results so far are in `results/nirai_overlap_seeder_*.txt` and `results/nirai_triage_seeder_payoff_melb2.txt`. Not final: the crowd-filter arms, the assembly check and the README write-up are still running; numbers in the README may change. Only tested on melb2 and (zero-shot) KPC2.
