#!/bin/bash
# Run stage B (niced) for each flow cell as soon as its basecalling is logged as done in prep.log.
S=/home/sgomathi/thusumbu/experiments/experiment_038_rawsamble_replication/scripts; L=/home/sgomathi/rawsamble_work/kpc2_fc/prep.log
for FC in FAX39617 FAX39616 FAV69578; do
  until grep -q "$FC: basecall done" $L; do sleep 60; done
  nice -n 10 $S/kpc2_fc_stageB.sh $FC > /home/sgomathi/rawsamble_work/kpc2_fc/$FC/stageB.log 2>&1
done
echo STAGEB_ALL_DONE >> /home/sgomathi/rawsamble_work/kpc2_fc/stageB_watch.log
