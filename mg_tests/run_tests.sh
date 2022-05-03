#! /usr/bin/env bash

DEFAULT_CONFIG_FILE_NAME=test_euler.ini

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=${SCRIPT_DIR}/..
GEF_MAIN=main_gef.py
CONFIG_FILE_BASE=${1:-${SCRIPT_DIR}/${DEFAULT_CONFIG_FILE_NAME}}
CONFIG_DIR=${SCRIPT_DIR}/configs
CONFIG_FILE=test.ini

test -f ${CONFIG_FILE_BASE} || CONFIG_FILE_BASE=${SCRIPT_DIR}/${DEFAULT_CONFIG_FILE_NAME}

echo "Base config file is ${CONFIG_FILE_BASE}"

cp ${CONFIG_FILE_BASE} ${CONFIG_FILE}

# MPIRUN="aprun"
MPIRUN="mpirun"

cd ${SCRIPT_DIR}
mkdir -p ${CONFIG_DIR}

function run_program() {
    if [ $# -gt 0 ]; then
        the_config=${1}
    else
        the_config=${CONFIG_FILE}
    fi
    set -x
    ${MPIRUN} -n 6 python3 ${GEF_DIR}/${GEF_MAIN} ${the_config}
    set +x
}

function set_parameters() {
    sed -i ${CONFIG_FILE} -e 's/^preconditioner *= *.*$/preconditioner = '${1}'/' \
                          -e 's/^nbsolpts *= *.*$/nbsolpts = '${2}'/' \
                          -e 's/^nb_elements_horizontal *= *.*$/nb_elements_horizontal = '${3}'/' \
                          -e 's/^precond_tolerance *= *.*$/precond_tolerance = '${4}'/' \
                          -e 's/^coarsest_mg_order *= *.*$/coarsest_mg_order = '${5}'/' \
                          -e 's/^mg_smoothe_only *= *.*$/mg_smoothe_only = '${6}'/' \
                          -e 's/^num_pre_smoothe *= *.*$/num_pre_smoothe = '${7}'/' \
                          -e 's/^num_post_smoothe *= *.*$/num_post_smoothe = '${8}'/' \
                          -e 's/^pseudo_cfl *= *.*$/pseudo_cfl = '${9}'/' \
                          -e 's/^dt *= *.*$/dt = '${10}'/' \
                          -e 's/^t_end *= *.*$/t_end = '${10}'/' \
                          -e 's/^dg_to_fv_interp *= *.*$/dg_to_fv_interp = '${11}'/' \
                          -e 's/^linear_solver *= *.*$/linear_solver = '${12}'/' \
                          -e 's/^mg_smoother *= *.*$/mg_smoother = '${13}'/' \
                          -e 's/^sgs_eta *= *.*$/sgs_eta = '${14}'/'
}

function compute_time_step() {
    if [ $# -lt 3 ]; then
        echo "Needs 3 arguments: the order, the number of elements, the reference time step"
        echo "The reference time step is for a problem of 2x30 (order 2, 30 elements)"
        exit
    fi

    order=$1
    num_el=$2

    factor=1.00
    if [ ${order} == 4 ]; then
        factor=0.66
        # factor=0.83
    elif [ ${order} == 8 ]; then
        factor=0.38
        # factor=0.69
    fi

    echo "${3} / ($order * $num_el) * $factor * 60" | bc | sed -e 's/\.[0-9]*//'
}

time_step=30
order=2
nb_elements=24

precond="none"
precond_tolerance=1e-1
coarsest_mg_order=1
mg_smoothe_only=1
num_pre_smoothe=1
num_post_smoothe=1

mg_smoother="erk"
sgs_eta=0.1

dg_to_fv_interp="lagrange"
# linear_solver="mg"
linear_solver="fgmres"

#################################
# 
pseudo_cfl=0.0039       # FV-MG RAT2
# pseudo_cfl=0.00156       # P-MG RAT2

rat2_fv_pseudo_cfl=0.0039
rat2_p_pseudo_cfl=0.00156
#################################

if [ "x${1}" == "x--gen-configs" ]; then

    ref_time_step=120
    orders="2 4"
    # orders=2
    # element_counts="30 60 120"
    element_counts="10 20"
    smoothings="1 2"
    # smoothings="4"
    mg_orders="1 2 4"
    pre_tols="1e-1 1e-6"
    preconds="fv-mg p-mg"
    #interps="lagrange l2-norm"
    interps="lagrange"
    for order in ${orders}; do
        for nb_elements in ${element_counts}; do
            [ $nb_elements -gt 60 ] && [ $order -gt 2 ] && continue
            [ $nb_elements -gt 30 ] && [ $order -gt 4 ] && continue

            # time_step=$(compute_time_step ${order} ${nb_elements} ${ref_time_step})

            echo "Time step: ${time_step}"

            precond="none"
            linear_solver="fgmres"
            set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${coarsest_mg_order} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${pseudo_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${sgs_eta}
            cp ${CONFIG_FILE} "${CONFIG_DIR}/config.a${precond}_${order}_${nb_elements}_${time_step}_${linear_solver}.ini"

            # linear_solver="multigrid"
            # coarsest_mg_order=${order}
            # num_pre_smoothe=1
            # num_post_smoothe=1
            # # for coarsest_mg_order in 1 2 4 8; do
            #     # for sm in 2 5 10; do
            #         # num_pre_smoothe=${sm}
            #         # num_post_smoothe=${sm}
            #         set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${coarsest_mg_order} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${pseudo_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${sgs_eta}
            #         cp ${CONFIG_FILE} "${CONFIG_DIR}/config.b${precond}_${order}_${nb_elements}_${time_step}_${linear_solver}_${coarsest_mg_order}_${mg_smoother}_${num_pre_smoothe}_${num_post_smoothe}.ini"
            #     # done
            # # done

            linear_solver="fgmres"
            precond="fv"
            for precond_tolerance in ${pre_tols}; do
                for dg_to_fv_interp in ${interps}; do
                    set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${coarsest_mg_order} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${pseudo_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${sgs_eta}
                    cp ${CONFIG_FILE} "${CONFIG_DIR}/config.z${precond}_${order}_${nb_elements}_${dg_to_fv_interp:0:3}_${precond_tolerance}_${time_step}.ini"
                done
            done

            precond="fv-mg"
            dg_to_fv_interp="lagrange"
            for precond in ${preconds}; do
            for coarsest_mg_order in ${mg_orders}; do
                [ $coarsest_mg_order -gt $order ] && continue
                for mg_smoothe_only in 1; do
                for num_pre_smoothe in ${smoothings}; do
                    num_post_smoothe=${num_pre_smoothe}
                    set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${coarsest_mg_order} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${pseudo_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${sgs_eta}
                    cp ${CONFIG_FILE} "${CONFIG_DIR}/config.c${precond}_${order}_${nb_elements}_${precond_tolerance}_${coarsest_mg_order}_${mg_smoother}_${mg_smoothe_only}_${num_pre_smoothe}_${num_post_smoothe}_${pseudo_cfl}_${time_step}_${dg_to_fv_interp:0:3}.ini"
                done
                done
            done
            done
        done
    done

elif [ "x${1}" == "x--run-configs" ]; then
    export PYTHONPATH=${PYTHONPATH}:${GEF_DIR}

    for config in $(ls ${CONFIG_DIR}); do
        echo "Running ${config}"
        run_program ${CONFIG_DIR}/${config} && rm -v ${CONFIG_DIR}/${config} || exit
    done
else
    set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${coarsest_mg_order} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${pseudo_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${sgs_eta}
    if [ "x${1}" != "x--just-config" ]; then
        run_program
    fi
fi
