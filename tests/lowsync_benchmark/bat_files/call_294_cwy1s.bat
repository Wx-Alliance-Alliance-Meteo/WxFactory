#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh


methods=("srerk6/galewsky_cwy1s.ini" "srerk6/case6_cwy1s.ini" "srerk6/case5_cwy1s.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 294 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
  done
done
