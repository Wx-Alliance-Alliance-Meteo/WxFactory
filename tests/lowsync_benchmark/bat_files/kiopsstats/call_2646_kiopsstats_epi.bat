#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./load_env.sh

methods=("epi2/case5_kiops.ini" "epi2/case5_pmex1s.ini" "epi2/case5_pmexne.ini" "epi2/case5_cwy1s.ini" "epi2/case5_cwyne.ini" "epi2/case5_icwy1s.ini" "epi2/case5_icwyne.ini" "epi4/case5_kiops.ini" "epi4/case5_pmex1s.ini" "epi4/case5_pmexne.ini" "epi4/case5_cwy1s.ini" "epi4/case5_cwyne.ini" "epi4/case5_icwy1s.ini" "epi4/case5_icwyne.ini" "epi6/case5_kiops.ini" "epi6/case5_pmex1s.ini" "epi6/case5_pmexne.ini" "epi6/case5_cwy1s.ini" "epi6/case5_cwyne.ini" "epi6/case5_icwy1s.ini" "epi6/case5_icwyne.ini" "epi2/case6_kiops.ini" "epi2/case6_pmex1s.ini" "epi2/case6_pmexne.ini" "epi2/case6_cwy1s.ini" "epi2/case6_cwyne.ini" "epi2/case6_icwy1s.ini" "epi2/case6_icwyne.ini" "epi4/case6_kiops.ini" "epi4/case6_pmex1s.ini" "epi4/case6_pmexne.ini" "epi4/case6_cwy1s.ini" "epi4/case6_cwyne.ini" "epi4/case6_icwy1s.ini" "epi4/case6_icwyne.ini" "epi6/case6_kiops.ini" "epi6/case6_pmex1s.ini" "epi6/case6_pmexne.ini" "epi6/case6_cwy1s.ini" "epi6/case6_cwyne.ini" "epi6/case6_icwy1s.ini" "epi6/case6_icwyne.ini" "epi2/galewsky_kiops.ini" "epi2/galewsky_pmex1s.ini" "epi2/galewsky_pmexne.ini" "epi2/galewsky_cwy1s.ini" "epi2/galewsky_cwyne.ini" "epi2/galewsky_icwy1s.ini" "epi2/galewsky_icwyne.ini" "epi4/galewsky_kiops.ini" "epi4/galewsky_pmex1s.ini" "epi4/galewsky_pmexne.ini" "epi4/galewsky_cwy1s.ini" "epi4/galewsky_cwyne.ini" "epi4/galewsky_icwy1s.ini" "epi4/galewsky_icwyne.ini" "epi6/galewsky_kiops.ini" "epi6/galewsky_pmex1s.ini" "epi6/galewsky_pmexne.ini" "epi6/galewsky_cwy1s.ini" "epi6/galewsky_cwyne.ini" "epi6/galewsky_icwy1s.ini" "epi6/galewsky_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
done
