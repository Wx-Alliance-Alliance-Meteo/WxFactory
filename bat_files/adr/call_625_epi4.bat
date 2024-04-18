#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


#methods=("kiops" "pmex_1s" "pmex" "cwy_1s" "cwy_ne" "icwy_1s" "icwy_ne")

methods=("pmex_ne1s" "icwy_ne1s" "cwy_ne1s")
integrators=("epi4" "epi5" "epi6")

intlen=${#integrators[@]}
methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((n=0; n < $intlen; n++)); do
  for ((k=0; k < $methodlen; k++)); do
     for ((j = 0; j < 7; j++)); do
       mpirun -np 625 python3 ./main_adr.py ${integrators[${n}]} ${methods[${k}]}
    done
  done
done
