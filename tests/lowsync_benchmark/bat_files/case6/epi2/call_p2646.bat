#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh

methods=("case6_kiops.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_cwyne1s.ini" "case6_icwy1s.ini" "case6_icwyne.ini" "case6_icwyne1s.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case6_pmexne1s.ini" "case6_kiops_ne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/epi2/${methods[${k}]}
   done
done
