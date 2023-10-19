#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh


methods=("cwy_ne" "icwy_1s" "icwy_ne" "cwy_1s")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 100 python3 ./main_adr.py 4 ${methods[${k}]}
  done
done
