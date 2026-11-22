#!/bin/bash
# Phase 1 (cheap, signal-only) for the other KPC2 flow cells: merge+time-sort POD5, noise features, online overlaps, reference mapping, truth positions.
# Phase 2 (slow): Dorado hac CPU basecalling of the first 8000 reads of each flow cell.   usage: kpc2_fc_prep.sh FC1 FC2 ...
B=/home/sgomathi/rawsamble_work; S=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/scripts
M=$B/src/RawHash/extern/local_kmer_models/uncalled_r1041_model_only_means.txt
REF=$B/kpc2_truth/KPC2__202310/reference.fna; DOR=$B/dorado_dl/bin/bin/dorado; MOD=$B/dorado_dl/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
for FC in "$@"; do
  R=$B/kpc2_fc/$FC; mkdir -p $R/stream $R/run $R/ds; cd $R
  echo "[$(date +%T)] $FC: sort"; python3 $S/make_sorted_pod5.py stream/sorted.pod5 stream/order.tsv $B/kpc2_run/raw/Klebsiella_pneumoniae_KPC2/${FC}_pass_*.pod5 > stream/sort.log 2>&1
  echo "[$(date +%T)] $FC: features"; python3 $S/best_read_features.py stream/sorted.pod5 stream/sigfeat.tsv
  echo "[$(date +%T)] $FC: online overlaps"; /usr/bin/time -f "wall %e s maxrss %M KB" -o run/online.time $B/rawhash2_triage -x ava-sensitive --r10 --online -p $M -t 8 stream/sorted.pod5 > run/online.paf 2> run/online.log
  echo "[$(date +%T)] $FC: reference mapping"; $B/rawhash2_noname -x sensitive --r10 -p $M -t 6 $REF stream/sorted.pod5 > run/map_ref2.paf 2> run/map_ref2.log
  python3 $S/kpc2_mk_truth.py $R
  tail -n +2 stream/order.tsv | head -8000 | cut -f1 > stream/ids_first8000.txt
  pod5 filter stream/sorted.pod5 --ids stream/ids_first8000.txt --output stream/first8000.pod5 --force-overwrite -t 8 > /dev/null 2>&1
  echo "[$(date +%T)] $FC: phase 1 done: $(wc -l < stream/order.tsv) reads"
done
echo PHASE1_DONE
for FC in "$@"; do
  R=$B/kpc2_fc/$FC; cd $R
  echo "[$(date +%T)] $FC: basecalling first 8000"; $DOR basecaller $MOD stream/first8000.pod5 --device cpu --emit-fastq > stream/first8000_hac.fastq 2> stream/first8000_hac.err
  echo "[$(date +%T)] $FC: basecall done"
done
echo ALL_DONE
