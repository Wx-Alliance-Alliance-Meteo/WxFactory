#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("case5_pmex1s.ini" "case6_pmex1s.ini" "galewsky_pmex1s.ini" "case5_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 1176 python3 ./main_gef.py config/test_files/srerk6/${methods[${k}]}
  done
done