#!/usr/bin/env bash
cd /home/vma000/code/gef_tanya
. ./load_env.sh


#methods=("epi4/case5_pmex1s.ini" "epi4/case5_cwy1s.ini" "epi4/case5_icwy1s.ini" "epi4/case5_kiops.ini" "epi4/case6_pmex1s.ini" "epi4/case6_cwy1s.ini" "epi4/case6_icwy1s.ini" "epi4/case6_kiops.ini" "epi5/case5_pmex1s.ini" "epi5/case5_cwy1s.ini" "epi5/case5_icwy1s.ini" "epi5/case5_kiops.ini" "epi5/case6_pmex1s.ini" "epi5/case6_cwy1s.ini" "epi5/case6_icwy1s.ini" "epi5/case6_kiops.ini" "epi6/case5_pmex1s.ini" "epi6/case5_cwy1s.ini" "epi6/case5_icwy1s.ini" "epi6/case5_kiops.ini" "epi6/case6_pmex1s.ini" "epi6/case6_cwy1s.ini" "epi6/case6_icwy1s.ini" "epi6/case6_kiops.ini" "epi4/galewsky_pmex1s.ini" "epi4/galewsky_cwy1s.ini" "epi4/galewsky_icwy1s.ini" "epi4/galewksy_kiops.ini" "epi5/galewsky_pmex1s.ini" "epi5/galewsky_cwy1s.ini" "epi5/galewsky_icwy1s.ini" "galewsky_kiops.ini" "epi6/galewsky_pmex1s.ini" "epi6/galewsky_cwy1s.ini" "epi6/galewsky_icwy1s.ini" "epi6/galewsky_kiops.ini")


methods=("epi4/case5_kiops.ini" "epi4/case6_kiops.ini" "epi5/case5_kiops.ini" "epi5/case6_kiops.ini" "epi6/case5_kiops.ini" "epi6/case6_kiops.ini" "epi4/galewksy_kiops.ini" "epi5/galewsky_pmex1s.ini" "epi5/galewsky_cwy1s.ini" "epi5/galewsky_icwy1s.ini" "epi5/galewsky_kiops.ini" "epi6/galewsky_kiops.ini")

methodlen=${#methods[@]}
#looping through each case and running it 7 times
for ((k=0; k < $methodlen; k++)); do
   for ((j = 0; j < 7; j++)); do
     mpirun -np 10584 python3 ./main_gef.py config/test_files/${methods[${k}]}
  done
done
