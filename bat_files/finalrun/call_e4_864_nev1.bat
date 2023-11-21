#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi4/case5_cwyne.ini" "epi4/case5_pmexne.ini" "epi4/case5_icwyne.ini" "epi4/case5_kiops.ini""epi4/case6_kiops.ini" "epi4/case6_pmexne.ini" "epi4/case6_cwyne.ini" "epi4/case6_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
