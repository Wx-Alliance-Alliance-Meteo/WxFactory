#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("kiops" "pmex_1s" "pmex" "cwy_1s" "cwy_ne" "icwy_1s" "icwy_ne")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 100 python3 ./main_ac.py srerk6 ${methods[${k}]}
  done
done
