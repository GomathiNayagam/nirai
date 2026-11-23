#!/bin/bash
# usage: run_chunk.sh NAME   (NAME.pod5 -> NAME.fastq, hac basecalls, low priority)
cd ~/rawsamble_work/rdh275/moredata
nice -n 10 ~/rawsamble_work/dorado_dl/bin/bin/dorado basecaller ~/rawsamble_work/dorado_dl/models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0 $1.pod5 --device cpu --emit-fastq > $1.fastq.tmp 2> $1.err && mv $1.fastq.tmp $1.fastq && echo done > $1.flag
