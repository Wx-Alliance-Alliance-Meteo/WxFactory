
methods=("call_p294_og.bat" "call_p294_cwy1s.bat" "call_p294_cwyne.bat" "call_p294_cwyne1s.bat" "call_p294_icwy1s.bat" "call_p294_icwyne.bat" "call_p294_icwy_ne1s.bat" "call_p294_icwyiop.bat" "call_p294_pmex1s.bat" "call_p294_pmexne.bat" "call_p294_pmexne1s.bat" "call_p294_kiopsne.bat")
methodlen=${#methods[@]}

#for case 5
for ((k = 0; k $methodlen; k++)); do
     ord_soumet -cpus 294 -jn gef_294_c5 -mpi -clone 7 -share e bat_files/case5/${methods[$k]}
done

#for case 6
for ((k = 0; k $methodlen; k++)); do
     ord_soumet -cpus 294 -jn gef_294_c6 -mpi -clone 7 -share e bat_files/case6/${methods[$k]}
done

#for case 8
for ((k = 0; k $methodlen; k++)); do
     ord_soumet -cpus 294 -jn gef_294_c8 -mpi -clone 7 -share e bat_files/case8/${methods[$k]}
done


