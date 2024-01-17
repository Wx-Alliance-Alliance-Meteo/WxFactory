#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("pmex" "icwy_ne" "cwy_ne")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 625 python3 ./main_burg.py srerk3 ${methods[${k}]}
  done
done
