#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("case5_kiops.ini" "case6_kiops.ini" "galewsky_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
  for ((j = 0; j < 7; j++)); do
       mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/epi4/${methods[${k}]}
  done
done
