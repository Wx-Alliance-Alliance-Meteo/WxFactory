#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("case6_pmexne.ini" "case6_icwyne.ini" "case5_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 294 python3 ./main_gef.py config/test_files/srerk3/${methods[${k}]}
  done
done
