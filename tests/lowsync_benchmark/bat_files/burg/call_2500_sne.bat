#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("kiops" "pmex_1s" "icwy_1s" "cwy_1s" "pmex_ne1s" "icwy_ne1s" "cwy_ne1s" "pmex" "cwy_ne" "icwy_ne")
integrators=("srerk3" "srerk6")

methodlen=${#methods[@]}
intlen=${#integrators[@]}
#looping through each case and running it 7 times
for ((n=0; n < $intlen; n++)); do 
  for ((k=0; k < $methodlen; k++)); do
     for ((j = 0; j < 7; j++)); do
       mpirun -np 2500 python3 ./tests/lowsync_benchmark/main_burg.py ${integrators[${n}]} ${methods[${k}]}
    done
  done
done
