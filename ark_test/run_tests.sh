#!/usr/bin/env bash

case_file="test_case.ini"

RUN="/usr/bin/mpirun -H 127.0.0.1:6 -merge-stderr-to-stdout -tag-output -n 6 -cpu-list 0,1,1,2,2,3 python3 ../main_gef.py ${case_file}"

SOL_PT_COUNTS="4 5"
ELEM_COUNTS="10 25"
TIME_STEPS="450 1800"

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

#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ARK4(3)7L[2]SA-ERK/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ARK4(3)7L[2]SA-ESDIRK/' \
#
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

#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ARK5(4)8L[2]SA-ERK/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ARK5(4)8L[2]SA-ESDIRK/' \
#
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
#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ARK5(4)8L[2]SAb-ERK/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ARK5(4)8L[2]SAb-ESDIRK/' \
#
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

#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ARK3(2)4L[2]SA-ERK/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ARK3(2)4L[2]SA-ESDIRK/' \
#
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

#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK3(2)5L[2]SA/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK3(2)5L[2]SA/' \
#
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
#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK3(2I)5L[2]SA/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK3(2I)5L[2]SA/' \
#
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
#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK4(3)6L[2]SA/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK4(3)6L[2]SA/' \
#
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
#sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK4(3I)6L[2]SA/' \
#                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK4(3I)6L[2]SA/' \
#
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

sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=QESDIRK4(3)6L[2]SA/' \
                    -e 's/^ark_solver_imp=.*/ark_solver_imp=QESDIRK4(3)6L[2]SA/' \

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

sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK5(3)6L[2]SA/' \
                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK5(3)6L[2]SA/' \

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

sed -i ${case_file} -e 's/^ark_solver_exp=.*/ark_solver_exp=ESDIRK5(4)7L[2]SA/' \
                    -e 's/^ark_solver_imp=.*/ark_solver_imp=ESDIRK5(4)7L[2]SA/' \

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


