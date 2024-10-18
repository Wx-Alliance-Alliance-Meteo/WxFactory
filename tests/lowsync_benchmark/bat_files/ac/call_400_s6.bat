#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


#methods=("kiops" "pmex_1s" "icwy_1s" "cwy_1s")
methods=("pmex" "icwy_ne" "cwy_ne")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 400 python3 ./tests/lowsync_benchmark/main_ac.py srerk6 ${methods[${k}]}
  done
done
