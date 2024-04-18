#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("pmex" "icwy_ne" "cwy_ne" "cwy_1s")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 625 python3 ./main_ac.py epi6 ${methods[${k}]}
  done
done
