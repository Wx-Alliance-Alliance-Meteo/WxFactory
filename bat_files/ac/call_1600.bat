#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("pmex_ne1s" "cwy_ne1s" "icwy_ne1s")
integrators=("epi4" "epi5" "epi6" "srerk3" "srerk6")

methodlen=${#methods[@]}
intlen=${#integrators[@]}
#looping through each case and running it 7 times
for ((n=0; n < $intlen; n++)); do 
  for ((k=0; k < $methodlen; k++)); do
     for ((j = 0; j < 7; j++)); do
       mpirun -np 1600 python3 ./main_adr.py ${integrators[${n}]} ${methods[${k}]}
    done
  done
done
