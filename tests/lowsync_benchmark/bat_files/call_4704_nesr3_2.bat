#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh


methods=("srerk3/case5_pmexne.ini" "srerk3/case6_pmexne.ini" "srerk3/galewsky_pmexne.ini" "srerk3/galewsky_icwyne.ini" "srerk3/case6_icwyne.ini" "srerk3/case5_icwyne.ini" "srerk3/case6_cwyne.ini" "srerk3/case5_cwyne.ini" "srerk3/galewsky_cwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 4704 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
