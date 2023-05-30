#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh

methods=("galewsky_cwyne.ini" "galewsky_pmexne1s.ini", "galewsky_kiops.ini" "case5_pmexne1s.ini" "case6_pmexne1s.ini")


methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/epi4/${methods[${k}]}
   done
done
