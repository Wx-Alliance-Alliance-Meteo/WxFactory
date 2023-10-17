#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh


methods=("srerk6/case5_pmexne.ini" "srerk6/case5_cwyne.ini" "srerk6/case5_icwyne.ini" "srerk3/case5_pmexne.ini" "srerk3/case5_cwyne.ini" "srerk3/case6_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 1176 python3 ./main_gef.py config/${methods[${k}]}
  done
done
