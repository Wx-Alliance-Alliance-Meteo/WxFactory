#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("epi5/case5_kiops.ini" "epi5/case6_kiops." "epi5/galewsky_kiops.ini" "epi5/case5_pmex1s.ini" "epi5/case6_pmex1s.ini" "epi5/galewsky_pmex1s.ini" "epi5/case5_icwy1s.ini" "epi5/case6_icwy1s.ini" "epi5/galewsky_icwy1s.ini" "epi5/case5_cwy1s.ini" "epi5/case6_cwy1s.ini" "epi5/galewsky_cwy1s.ini" "epi5/case5_pmexne.ini" "epi5/case6_pmexne.ini" "epi5/galewsky_pmexne.ini" "epi5/case5_icwyne.ini" "epi5/case6_icwyne.ini" "epi5/galewsky_icwyne.ini" "epi5/case6_cwyne.ini" "epi5/case5_cwyne.ini" "epi5/galewsky_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 4704 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
