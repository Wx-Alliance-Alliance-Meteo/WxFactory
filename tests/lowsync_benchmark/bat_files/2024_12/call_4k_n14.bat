#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("galewsky_k_og_n14.ini" "galewsky_pmx_og_n14.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 3; j++)); do
     mpirun -np 4704 python3 ./WxFactory tests/lowsync_benchmark/config/2024_12/${methods[${k}]}
  done
done
