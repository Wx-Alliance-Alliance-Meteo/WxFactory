#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi6/case5_cwyne.ini" "epi6/case5_pmexne.ini" "epi6/case5_icwyne.ini" "epi6/case6_pmexne.ini" "epi6/case6_cwyne.ini" "epi6/case6_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
