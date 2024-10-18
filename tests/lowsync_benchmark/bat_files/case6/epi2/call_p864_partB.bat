#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./load_env.sh

methods=("case6_icwyne1s.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case6_pmexne1s.ini", "case6_kiops_ne.ini")

#methods=("case6_kiopsne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/epi2/${methods[${k}]}
   done
done
