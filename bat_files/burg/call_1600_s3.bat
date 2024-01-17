#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("kiops" "pmex_1s" "cwy_1s" "icwy_1s" "icwy_ne1s" "pmex_ne1s" "cwy_ne1s" "pmex" "icwy_ne" "cwy_ne")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 1600 python3 ./main_burg.py srerk3 ${methods[${k}]}
  done
done
