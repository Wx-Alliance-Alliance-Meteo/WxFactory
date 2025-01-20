#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("case5_pmex1s.ini" "case5_cwy1s.ini" "case5_icwy1s.ini" "case6_pmex1s.ini" "case6_cwy1s.ini" "case6_icwy1s.ini" "galewsky_pmex1s.ini" "galewsky_cwy1s.ini" "galewsky_icwy1s.ini" "galewsky_kiops.ini" "case5_kiops.ini" "case6_kiops.ini" "galewsky_pmexne.ini" "galewsky_cwyne.ini" "galewsky_icwyne.ini" "case5_pmexne.ini" "case6_pmexne.ini" "case5_icwyne.ini" "case6_icwyne.ini" "case5_cwyne.ini" "case6_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 10584 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk3/${methods[${k}]}
  done
done
