#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


#methods=("kiops" "pmex_1s" "icwy_1s" "cwy_1s")
methods=("pmex" "icwy_ne" "cwy_ne")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 400 python3 ./main_ac.py srerk3 ${methods[${k}]}
  done
done
