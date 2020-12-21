#!/usr/bin/env bash

case_file="test_case.ini"

RUN="/usr/bin/mpirun -H 127.0.0.1:6 -merge-stderr-to-stdout -tag-output -n 6 -cpu-list 0,1,1,2,2,3 python3 ../main_gef.py ${case_file}"

SOL_PT_COUNTS="4 5"
ELEM_COUNTS="15 30"
TIME_STEPS="1800"

SCHEMES_BOTH="ARK4(3)7L[2]SA ARK5(4)8L[2]SA ARK5(4)8L[2]SAb ARK3(2)4L[2]SA"
#SCHEMES_IMP="ESDIRK3(2)5L[2]SA ESDIRK3(2I)5L[2]SA ESDIRK4(3)6L[2]SA ESDIRK4(3I)6L[2]SA ESDIRK4(3I)6L[2]SA QESDIRK4(3)6L[2]SA ESDIRK5(3)6L[2]SA ESDIRK5(4)7L[2]SA"

#for nb_sol_pt in ${SOL_PT_COUNTS}; do
#    for nb_elem in ${ELEM_COUNTS}; do
#        for dt in ${TIME_STEPS}; do
#            sed -i ${case_file} -e 's/^nbsolpts=.*/nbsolpts='${nb_sol_pt}'/' \
#                                -e 's/^nb_elements=.*/nb_elements='${nb_elem}'/' \
#                                -e 's/^dt=.*/dt='${dt}'/'
#            ${RUN}
#        done
#    done
#done
#


for scheme in ${SCHEMES_BOTH}; do
    sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp='${scheme}'-ERK/' \
                        -e 's/^ark_solver_imp=.*/ark_solver_imp='${scheme}'-ESDIRK/' \

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
done

for scheme in ${SCHEMES_IMP}; do
    sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp='${scheme}'/' \
                        -e 's/^ark_solver_imp=.*/ark_solver_imp='${scheme}'/' \

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
done

