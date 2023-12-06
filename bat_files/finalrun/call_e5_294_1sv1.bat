#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi5/case5_cwy1s.ini" "epi5/case5_pmex1s.ini" "epi5/case5_icwy1s.ini" "epi5/case5_kiops.ini" "epi5/case6_kiops.ini" )

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 294 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
