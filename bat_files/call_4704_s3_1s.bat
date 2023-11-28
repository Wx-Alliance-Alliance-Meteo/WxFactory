#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("case5_kiops.ini" "case5_pmex1s.ini" "case5_icwy1s.ini" "case5_cwy1s.ini" "case6_kiops.ini" "case6_pmex1s.ini" "case6_icwy1s.ini" "case6_cwy1s.ini" "galewsky_kiops.ini" "galewsky_pmex1s.ini" "galewsky_icwy1s.ini" "galewsky_cwy1s.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 4704 python3 ./main_gef.py config/test_files/srerk3/${methods[${k}]}
  done
done
