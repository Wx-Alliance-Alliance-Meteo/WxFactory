#!/usr/bin/env bash
cd ${WX_FACTORY_DIR}
. ./scripts/load_env.sh
mpirun -np 294 python3 ./WxFactory config/procs294/case5_cwy1s.ini
