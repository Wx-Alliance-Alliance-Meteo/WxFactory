#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("srerk6/case5_pmexne.ini" "srerk6/case5_cwyne.ini" "srerk6/case5_icwyne.ini" "srerk6/case6_pmexne.ini" "srerk6/case6_cwyne.ini" "srerk6/case6_icwyne.ini" "srerk6/galewsky_pmexne.ini" "srerk6/galewsky_icwyne.ini" "srerk6/galewsky_cwyne.ini" "srerk3/galewsky_kiops.ini" "srerk3/case5_kiops.ini" "srerk3/case6_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 10584 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done