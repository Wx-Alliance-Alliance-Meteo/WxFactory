#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("pmex" "cwy_ne" "icwy_ne")
integrator=("srerk3" "srerk6")

methodlen=${#methods[@]}
intlen=${#integrator[@]}

#looping through each case and running it 7 times
for ((n=0; n < $intlen; n++)); do
  for ((k=0; k < $methodlen; k++)); do
     for ((j = 0; j < 7; j++)); do
       mpirun -np 625 python3 ./main_ac.py ${integrator[${n}]} ${methods[${k}]}
    done
  done
done