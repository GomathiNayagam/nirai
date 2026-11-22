#!/bin/bash
# Stage C: when all basecalling, stage B and the pooled overlaps are done, build pooled.fastq and run the pooled-depth stopping test.
B=/home/sgomathi/rawsamble_work; S=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/scripts
until grep -q ALL_DONE $B/kpc2_fc/prep.log && grep -q STAGEB_ALL_DONE $B/kpc2_fc/stageB_watch.log && grep -q POOL_OVERLAP_DONE $B/kpc2_pool/build.log; do sleep 120; done
cd $B/kpc2_pool; mkdir -p ds
cat $B/kpc2_run/stream/first8000_hac.fastq $B/kpc2_fc/FAX39617/stream/first8000_hac.fastq $B/kpc2_fc/FAX39616/stream/first8000_hac.fastq $B/kpc2_fc/FAV69578/stream/first8000_hac.fastq > pooled.fastq
echo "[$(date +%T)] stage C start: $(grep -c '^+$' pooled.fastq) fastq records (approx)"
python3 -W ignore $S/pooled_depth_test.py > /home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/results/kpc2_pooled_depth_stopping_test.txt 2> $B/kpc2_pool/stageC.err
echo "[$(date +%T)] stage C done"
