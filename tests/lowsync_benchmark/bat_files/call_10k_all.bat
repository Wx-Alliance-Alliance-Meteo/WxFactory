#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("srerk6/galewsky_pmex1s.ini" "srerk6/case6_icwy1s.ini" "srerk6/case5_pmexne.ini" "srerk6/case6_pmexne.ini" "srerk6/galewsky_pmexne.ini" "srerk6/case5_icwyne.ini" "srerk6/case6_icwyne.ini" "srerk6/galewsky_icwyne.ini" "srerk6/case5_cwyne.ini" "srerk6/case6_cwyne.ini" "srerk6/galewsky_cwyne.ini" "srerk3/case5_pmexne.ini" "srerk3/case6_pmexne.ini" "srerk3/galewsky_pmexne.ini" "srerk3/case5_cwyne.ini" "srerk3/case6_cwyne.ini" "srerk3/galewsky_cwyne.ini" "srerk3/case5_icwyne.ini" "srerk3/case6_icwy1s.ini" "srerk3/case6_icwy1s.ini" "epi6/galewsky_cwyne.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_pmexne.ini" "epi2/galewsky_pmex1s.ini" )

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 10584 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
