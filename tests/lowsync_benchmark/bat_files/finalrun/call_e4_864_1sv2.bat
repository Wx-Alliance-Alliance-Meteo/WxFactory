#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("epi4/galewsky_cwy1s.ini" "epi4/galewsky_pmex1s.ini" "epi4/galewsky_icwy1s.ini" "epi4/galewsky_kiops.ini" "epi4/galewsky_pmexne.ini" "epi4/galewsky_icwyne.ini" "epi4/galewsky_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 864 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
