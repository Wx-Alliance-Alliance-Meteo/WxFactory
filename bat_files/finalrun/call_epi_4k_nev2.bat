#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi4/galewsky_pmexne.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_icwyne.ini" "epi5/galewsky_pmexne.ini" "epi5/galewsky_cwyne.ini" "epi5/galewsky_icwyne.ini" "epi6/galewsky_pmexne.ini" "epi6/galewsky_cwyne.ini" "epi6/galewsky_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 4704 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
