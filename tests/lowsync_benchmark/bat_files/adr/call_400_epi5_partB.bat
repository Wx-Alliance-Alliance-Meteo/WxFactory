#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("cwy_ne" "icwy_1s" "icwy_ne" "cwy_1s")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 400 python3 ./tests/lowsync_benchmark/main_adr.py epi5 ${methods[${k}]}
  done
done
