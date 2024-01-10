#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh

methods=("srerk3/galewsky_cwy1s.ini" "srerk3/case6_cwy1s.ini" "srerk3/case5_cwy1s.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
