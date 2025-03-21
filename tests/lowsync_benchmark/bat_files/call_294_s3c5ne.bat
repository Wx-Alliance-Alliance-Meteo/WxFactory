#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh

#methods=("case5_pmexne.ini" "case5_icwyne.ini" "case5_cwyne.ini")
methods=("case5_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 294 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk3/${methods[${k}]}
  done
done
