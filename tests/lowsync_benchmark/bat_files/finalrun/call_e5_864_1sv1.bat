#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("epi5/case5_cwy1s.ini" "epi5/case5_pmex1s.ini" "epi5/case5_icwy1s.ini" "epi5/case5_kiops.ini" "epi5/case6_pmex1s.ini" "epi5/case6_cwy1s.ini" "epi5/case6_icwy1s.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 864 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
