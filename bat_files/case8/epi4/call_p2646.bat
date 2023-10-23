#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("galewksy_kiops.ini" "galewksy_cwy1s.ini" "galewksy_cwyne.ini" "galewksy_cwyne1s.ini" "galewksy_icwy1s.ini" "galewksy_icwyne.ini" "galewksy_icwyne1s.ini" "galewksy_icwyiop.ini" "galewksy_pmex1s.ini" "galewksy_pmexne.ini" "galewksy_pmexne1s.ini" "galewksy_kiops_ne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 2646 python3 ./main_gef.py config/test_files/epi4/${methods[${k}]}
   done
done
