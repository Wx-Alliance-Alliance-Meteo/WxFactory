#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


#methods=("epi4/case5_pmex1s.ini" "epi4/case5_cwy1s.ini" "epi4/case5_icwy1s.ini" "epi4/case5_kiops.ini" "epi4/case6_pmex1s.ini" "epi4/case6_cwy1s.ini" "epi4/case6_icwy1s.ini" "epi4/case6_kiops.ini" "epi4/galewsky_pmex1s.ini" "epi4/galewsky_cwy1s.ini" "epi4/galewsky_icwy1s.ini" "epi4/galewsky_kiops.ini" "epi4/case6_pmexne.ini" "epi4/case6_pmexne.ini" "epi4/galewsky_pmexne.ini" "epi4/case6_icwyne.ini" "epi4/case6_cwyne.ini" "epi4/case5_cwyne.ini" "epi4/case5_cwyne.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_icwyne.ini")

methods=("epi4/case5_pmexne.ini" "epi4/case5_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
