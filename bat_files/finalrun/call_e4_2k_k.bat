#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("case5_kiops.ini" "case6_kiops.ini" "galewsky_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((j = 0; j < 7; j++)); do
     mpirun -np 2646 python3 ./main_gef.py config/test_files/epi4/${methods[${k}]}
done
