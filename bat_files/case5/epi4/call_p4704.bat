#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh

methods=("case5_kiops.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_cwyne1s.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyne1s.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_pmexne1s.ini" "case5_kiops_ne.ini")
#methods=("case5.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_cwy1s.ini" "case5_icwy1s.ini" "case5_cwy1s_sm.ini" "case5_icwy1s_sm.ini")
methodlen=${#methods[@]}

#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 4704 python3 ./main_gef.py config/test_files/epi4/${methods[${k}]}
  done
done
