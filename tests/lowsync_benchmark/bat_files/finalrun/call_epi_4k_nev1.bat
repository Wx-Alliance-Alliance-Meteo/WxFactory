#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("epi6/case5_kiops.ini" "epi6/case6_kiops." "epi6/galewsky_kiops.ini" "epi6/case5_pmexne.ini" "epi6/case6_pmexne.ini" "epi6/galewsky_pmexne.ini" "epi6/case5_icwyne.ini" "epi6/case6_icwyne.ini" "epi6/galewsky_icwyne.ini" "epi6/case6_cwyne.ini" "epi6/case5_cwyne.ini" "epi6/galewsky_cwyne.ini" "epi5/galewsky_pmexne.ini" "epi5/galewsky_cwyne.ini" "epi5/galewsky_icwyne.ini" "epi5/galewsky_kiops.ini" "epi5/case6_kiops.ini" "epi5/case6_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 4704 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
