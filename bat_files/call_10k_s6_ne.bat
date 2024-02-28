#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("srerk6/galewsky_pmexne.ini" "srerk6/galewsky_icwyne.ini" "srerk6/galewsky_cwyne.ini" "srerk6/galewsky_kiops.ini" "srerk6/galewsky_pmex1s.ini" "srerk6/galewsky_cwy1s.ini" "srerk6/galewsky_icwy1s.ini" "srerk6/case6_pmexne.ini" "srerk6/case6_icwyne.ini" "srerk6/case6_cwyne.ini" "srerk6/case6_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 10584 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
