#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh
mpirun -np 294 python3 ./main_gef.py config/procs294/case5_cwy1s.ini
