#!/bin/bash
# hac basecalls for extra RDH275 reads so their mean Q can be used as training labels (never touches the gate/test sets)
cd ~/rawsamble_work/rdh275/moredata
for n in fail_all pass_extra; do
  nice -n 10 ~/rawsamble_work/dorado_dl/bin/bin/dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v6.0.0 $n.pod5 --device cpu --emit-fastq > $n.fastq.tmp 2> $n.err && mv $n.fastq.tmp $n.fastq
done; echo MORE_DONE > done.flag
