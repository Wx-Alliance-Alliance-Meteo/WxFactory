#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh

#looping through each case and running it 7 times
for ((j = 0; j < 7; j++)); do
   mpirun -np 294 python3 ./main_gef.py config/test_files/srerk6/galewski_pmexne.ini
done