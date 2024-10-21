#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh

#looping through each case and running it 7 times
for ((j = 0; j < 7; j++)); do
   mpirun -np 294 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/srerk6/case6_pmexne.ini
done
