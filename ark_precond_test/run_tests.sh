#!/usr/bin/env bash

case_file="test_case.ini"

RUN="/usr/bin/mpirun -H 127.0.0.1:6 -merge-stderr-to-stdout -tag-output -n 6 -cpu-list 0,1,1,2,2,3 python3 ../main_gef.py ${case_file}"

SOL_PT_COUNTS="4 5 6"
ELEM_COUNTS="10 15 20 25 30"
TIME_STEPS="450 900 1800 3600"

for nb_sol_pt in ${SOL_PT_COUNTS}; do
    for nb_elem in ${ELEM_COUNTS}; do
        for dt in ${TIME_STEPS}; do
            sed -i ${case_file} -e 's/^nbsolpts=.*/nbsolpts='${nb_sol_pt}'/' \
                                -e 's/^nb_elements=.*/nb_elements='${nb_elem}'/' \
                                -e 's/^dt=.*/dt='${dt}'/'
            ${RUN}
        done
    done
done


