#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("galewsky_k_og_n14.ini" "galewsky_k_og_n28.ini" "galewsky_pmx_og_n14.ini" "galewsky_pmx_og_n28.ini" "galewsky_pmx_og_dob.ini" "galewsky_k_og_dob.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 3; j++)); do
     mpirun -np 2646 python3 ./WxFactory config/time_config/${methods[${k}]}
  done
done
