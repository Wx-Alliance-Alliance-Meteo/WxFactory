#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


methods=("epi6/galewsky_cwyne.ini" "srerk3/galewsky_cwyne.ini" "epi6/galewsky_pmexne.ini" "epi6/galewsky_kiops.ini" "srerk3/galewsky_kiops.ini" "srerk3/galewsky_pmexne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
  for ((j = 0; j < 5; j++)); do
       mpirun -np 2646 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
