#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("epi6/galewsky_cwyne.ini" "epi6/case6_cwy1s.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_pmexne.ini" "epi2/galewsky_pmex1s.ini" "epi2/case6_cwyne.ini" "epi4/case6_cwyne.ini" "epi6/case6_cwyne.ini" "epi6/case5_cwy1s.ini")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 2646 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
