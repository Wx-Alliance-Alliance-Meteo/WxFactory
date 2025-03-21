#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("epi4/case6_cwyne.ini" "epi4/case6_pmexne.ini" "epi4/case6_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 294 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
