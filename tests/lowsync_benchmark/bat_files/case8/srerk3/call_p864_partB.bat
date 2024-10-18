#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh

methods=("galewsky_icwyne1s.ini" "galewsky_icwyiop.ini" "galewsky_pmex1s.ini" "galewsky_pmexne.ini" "galewsky_pmexne1s.ini", "galewsky_kiops_ne.ini")

#methods=("galewsky_kiopsne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk3/${methods[${k}]}
   done
done
