#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((j = 0; j < 7; j++)); do
   mpirun -np 294 python3 ./main_gef.py config/srerk6/galewski_pmex1s.ini
done
