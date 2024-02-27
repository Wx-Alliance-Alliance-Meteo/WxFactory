#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi5/galewsky_cwyne.ini" "epi5/galewsky_pmexne.ini" "epi5/galewsky_icwyne.ini" "epi5/galewsky_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
