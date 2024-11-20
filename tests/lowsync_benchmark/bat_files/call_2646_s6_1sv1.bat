#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh

methods=("case5_kiops.ini" "case6_kiops.ini" "galewsky_kiops.ini" "case5_pmex1s.ini" "case6_pmex1s.ini" "case5_icwy1s.ini" "case6_icwy1s.ini")


methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk6/${methods[${k}]}
  done
done
