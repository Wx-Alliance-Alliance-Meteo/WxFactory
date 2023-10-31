#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("srerk3/galewsky_pmex1s.ini" "srerk3/case6_icwy1s.ini" "srerk6/galewsky_pmex1s.ini" "srerk6/galewsky_icwy1s.ini" "srerk6/case6_cwy1s.ini" "srerk6/case6_icwy1s.ini" "srerk6/case6_kiops.ini" "epi4/case5_cwyne.ini" "epi2/case5_cwyne.ini")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 2646 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
