#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("epi6/case6_icwyne.ini" "epi6/case6_cwyne.ini" "epi6/case6_pmexne.ini" "epi6/case5_pmexne.ini" "epi6/case5_cwyne.ini" "epi6/case5_icwyne.ini" "epi6/case5_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 1176 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
