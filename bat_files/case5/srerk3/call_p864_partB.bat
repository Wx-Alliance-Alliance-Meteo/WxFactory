#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh

methods=("case5_icwyne1s.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_pmexne1s.ini", "case5_kiops_ne.ini")

#methods=("case5_kiopsne.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times

for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 864 python3 ./main_gef.py config/test_files/srerk3/${methods[${k}]}
   done
done
