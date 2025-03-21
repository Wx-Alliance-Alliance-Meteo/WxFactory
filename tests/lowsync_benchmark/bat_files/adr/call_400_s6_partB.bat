#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("cwy_ne1s" "icwy_ne1s" "pmex_ne1s")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 400 python3 ./tests/lowsync_benchmark/main_adr.py srerk6 ${methods[${k}]}
  done
done
