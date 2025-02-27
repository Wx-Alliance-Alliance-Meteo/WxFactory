#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


#methods=("epi5/galewsky_kiops.ini" "epi5/galewsky_pmexne.ini" "epi5/galewsky_cwyne.ini" "epi5/galewsky_icwyne.ini" "epi5/galewsky_case5_kiops.ini" "epi5/case6_kiops.ini")
methods=("epi5/case5_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 1176 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
