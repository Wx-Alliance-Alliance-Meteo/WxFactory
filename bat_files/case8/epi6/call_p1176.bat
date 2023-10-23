#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

#methods=("galewsky_kiops.ini" "galewsky_cwy1s.ini" "galewsky_cwyne.ini" "galewsky_cwyne1s.ini" "galewsky_icwy1s.ini" "galewsky_icwyne.ini" "galewsky_icwyne1s.ini" "galewsky_icwyiop.ini" "galewsky_pmex1s.ini" "galewsky_pmexne.ini" "galewsky_pmexne1s.ini" "galewsky_kiops_ne.ini")


methods=("galewsky_kiops.ini" "galewsky_cwyne.ini" "galewsky_kiops_ne.ini")

methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 1176 python3 ./main_gef.py config/test_files/epi6/${methods[${k}]}
   done
done
