#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("galewsky_k_adapt.ini" "galewsky_k_exp.ini" "galewsky_k_init.ini" "galewsky_k_insideortho.ini" "galewsky_k_og.ini" "galewsky_k_sol.ini" "galewsky_k_ortho.ini" "galewsky_pmx_adapt.ini" "galewsky_pmx_exp.ini" "galewsky_pmx_init.ini" "galewsky_pmx_insideortho.ini" "galewsky_pmx_og.ini" "galewsky_pmx_sol.ini" "galewsky_pmx_ortho.ini")

methodlen=${#methods[@]}
for ((k=0; k < $methodlen; k++)); do
   mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/2024_12/${methods[${k}]}
done
