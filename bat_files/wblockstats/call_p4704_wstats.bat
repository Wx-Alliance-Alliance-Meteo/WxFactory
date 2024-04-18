#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh

methods=("case5_kiops.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case6_kiops.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_icwy1s.ini" "case6_icwyne.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "galewsky_kiops.ini" "galewsky_cwy1s.ini" "galewsky_cwyne.ini" "galewsky_icwy1s.ini" "galewsky_icwyne.ini" "galewsky_pmex1s.ini" "galewsky_pmexne.ini")

folders=("epi2" "srerk3" "epi4" "epi6" "srerk6")

methodlen=${#methods[@]}
folderlen=${#folders[@]}
for ((j=0; j < $folderlen; j++)); do
  for ((k=0; k < $methodlen; k++)); do
    mpirun -np 4704 python3 ./main_gef.py config/test_files/${folders[${j}]}/${methods[${k}]}
  done
done
