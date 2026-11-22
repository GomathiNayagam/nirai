#!/bin/bash
# Stage B for one flow cell once first8000_hac.fastq is complete.  usage: kpc2_fc_stageB.sh FC
FC=$1; B=/home/sgomathi/rawsamble_work; S=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/scripts
RES=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/results
M=$B/src/RawHash/extern/local_kmer_models/uncalled_r1041_model_only_means.txt; RH=$B/rawhash2_noname
export KROOT=$B/kpc2_fc/$FC; cd $KROOT; mkdir -p asm prefix
NB=$(grep -c '^+$' stream/first8000_hac.fastq); N=$(wc -l < stream/ids_first8000.txt); H=$((N/2))
echo "[$(date +%T)] $FC N=$N"
# 1) signal vs basecalled assembly
F=stream/first8000_hac.fastq
minimap2 -x ava-ont -t 8 $F $F > asm/ava_both.paf 2>/dev/null; minimap2 -x ava-ont --for-only -t 8 $F $F > asm/ava_for.paf 2>/dev/null
NS=$(( N>7000 ? 7979 : N-1 ))   # signal_assembly_test uses reads that were basecalled (n of fastq)
NBASE=$(grep -c '^@' $F)
python3 -I -W ignore $S/signal_assembly_test.py $((NBASE/8)),$((NBASE/4)),$((NBASE/2)),$((NBASE*3/4)),$NBASE > $RES/${FC}_signal_assembly_test.txt 2>&1
# 2) prefix-only rejection replay (default and looser settings)
python3 -I $S/prefix_build.py $H $N 2500,5000,10000 > prefix/build.log 2>&1
cd prefix; $RH -x ava-sensitive --r10 -t 16 -p $M -d hist.ind hist_$H.pod5 > /dev/null 2> idx.err
for K in 2500 5000 10000; do
  $RH -x ava-sensitive --r10 -t 4 hist.ind q_${H}_${N}_$K.pod5 > q_$K.paf 2>/dev/null
  $RH -x ava-sensitive --r10 -t 4 --min-score 8 --min-events 20 --min-anchors 2 hist.ind q_${H}_${N}_$K.pod5 > q_${K}_loose.paf 2>/dev/null
done
cd $KROOT
( python3 -I -W ignore $S/prefix_eval.py $H $N 2500,5000,10000; echo "######## looser chaining ########"; SUF=_loose python3 -I -W ignore $S/prefix_eval.py $H $N 5000,10000 ) > $RES/${FC}_prefix_rejection_replay.txt 2>&1
# 3) variant-calling policy comparison
python3 -W ignore $S/var_policies_kpc2.py $N real > $RES/${FC}_variant_downstream.txt 2>&1
echo "[$(date +%T)] $FC stage B done"
