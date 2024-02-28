#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("galewsky_pmex1s.ini" "galewsky_icwy1s.ini" "case5_cwy1s.ini" "case6_cwy1s.ini" "galewsky_cwy1s.ini")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 2646 python3 ./main_gef.py config/test_files/srerk6/${methods[${k}]}
  done
done
