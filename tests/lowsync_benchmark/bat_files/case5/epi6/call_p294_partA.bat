#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh

#methods=("case5.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_cwyne1s.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyne1s.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_pmexne1s.ini" "case5_kiopsne.ini")
methods=("case5_kiops.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_cwyne1s.ini")

methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 294 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/epi6/${methods[${k}]}
   done
done
