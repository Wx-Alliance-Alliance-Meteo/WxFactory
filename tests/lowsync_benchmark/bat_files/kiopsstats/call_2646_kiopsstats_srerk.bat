#!/usr/bin/env bash
cd /home/vma000/ords/gef_tanya
. ./scripts/load_env.sh

methods=("srerk3/case5_kiops.ini" "srerk3/case5_pmex1s.ini" "srerk3/case5_pmexne.ini" "srerk3/case5_cwy1s.ini" "srerk3/case5_cwyne.ini" "srerk3/case5_icwy1s.ini" "srerk3/case5_icwyne.ini" "srerk6/case5_kiops.ini" "srerk6/case5_pmex1s.ini" "srerk6/case5_pmexne.ini" "srerk6/case5_cwy1s.ini" "srerk6/case5_cwyne.ini" "srerk6/case5_icwy1s.ini" "srerk6/case5_icwyne.ini" "srerk3/case6_kiops.ini" "srerk3/case6_pmex1s.ini" "srerk3/case6_pmexne.ini" "srerk3/case6_cwy1s.ini" "srerk3/case6_cwyne.ini" "srerk3/case6_icwy1s.ini" "srerk3/case6_icwyne.ini" "srerk6/case6_kiops.ini" "srerk6/case6_pmex1s.ini" "srerk6/case6_pmexne.ini" "srerk6/case6_cwy1s.ini" "srerk6/case6_cwyne.ini" "srerk6/case6_icwy1s.ini" "srerk6/case6_icwyne.ini" "srerk3/galewsky_kiops.ini" "srerk3/galewsky_pmex1s.ini" "srerk3/galewsky_pmexne.ini" "srerk3/galewsky_cwy1s.ini" "srerk3/galewsky_cwyne.ini" "srerk3/galewsky_icwy1s.ini" "srerk3/galewsky_icwyne.ini" "srerk6/galewsky_kiops.ini" "srerk6/galewsky_pmex1s.ini" "srerk6/galewsky_pmexne.ini" "srerk6/galewsky_cwy1s.ini" "srerk6/galewsky_cwyne.ini" "srerk6/galewsky_icwy1s.ini" "srerk6/galewsky_icwyne.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   mpirun -np 2646 python3 ./WxFactory tests/lowsync_benchmark/config/test_files/${methods[${k}]}
done
