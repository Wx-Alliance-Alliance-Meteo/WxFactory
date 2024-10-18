#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh

methods=("galewsky_cwyne.ini" "galewsky_pmexne.ini" "galewsky_icwyne.ini" "galewsky_kiops.ini" "case6_kiops.ini" "case5_kiops.ini")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk6/${methods[${k}]}
  done
done
