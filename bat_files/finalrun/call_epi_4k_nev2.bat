#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("epi4/galewsky_pmexne.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_icwyne.ini" "epi4/galewsky_kiops.ini" "epi4/case5_pmexne.ini" "epi4/case5_icwyne.ini" "epi4/case5_cwyne.ini" "epi4/case5_kiops.ini" "epi4/case6_pmexne.ini" "epi4/case6_cwyne.ini" "epi4/case6_icwyne.ini" "epi4/case6_kiops.ini" "epi5/case5_pmexne.ini" "epi5/case5_cwyne.ini" "epi5/case5_icwyne.ini" "epi5/case5_kiops.ini" "epi5/case6_pmexne.ini" "epi5/case6_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 5; j++)); do
     mpirun -np 4704 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
