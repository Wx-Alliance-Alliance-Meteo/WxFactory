#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


#methods=("kiops" "pmex_1s" "pmex" "cwy_1s")
methods=("pmex_ne1s" "icwy_ne1s" "cwy_ne1s")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 100 python3 ./tests/lowsync_benchmark/main_adr.py epi4 ${methods[${k}]}
  done
done
