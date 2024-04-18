#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh

methods=("case6_kiops.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_cwyne1s.ini" "case6_icwy1s.ini" "case6_icwyne.ini")

#methods=("case6_kiopsne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/epi6/${methods[${k}]}
   done
done
