#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("srerk6/galewsky_pmexne.ini" "srerk6/galewsky_icwyne.ini" "srerk6/galewsky_cwyne.ini" "srerk6/galewsky_kiops.ini" "srerk6/case6_pmexne.ini" "srerk6/case6_icwyne.ini" "srerk6/case6_cwyne.ini" "srerk6/case6_kiops.ini" "srerk6/case5_pmexne.ini" "srerk6/case5_cwyne.ini" "srerk6/case5_icwyne.ini" "srerk6/case5_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 10584 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
