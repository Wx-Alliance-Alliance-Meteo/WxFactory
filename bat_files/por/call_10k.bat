#!/usr/bin/env bash
cd /home/vma000/code/wx_factory_tanya
. ./load_env.sh


methods=("kiops" "pmex_1s" "pmex" "cwy_1s" "cwy_ne" "icwy_1s" "icwy_ne")
order=(2 4 6)

methodlen=${#methods[@]}
orderlen=${#order[@]}

#looping through each case and running it 7 times
for ((n=0; n < $orderlen; n++)); do
  for ((k=0; k < $methodlen; k++)); do
     for ((j = 0; j < 7; j++)); do
       mpirun -np 10000 python3 ./main_por.py ${order[${n}]} ${methods[${k}]}
     done
  done
done
