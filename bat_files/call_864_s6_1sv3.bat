#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh

methods=("srerk6/galewsky_pmex1s.ini" "srerk6/galewsky_kiops.ini" "srerk6/galewsky_cwy1s.ini" "srerk6/galewsky_icwy1s.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
