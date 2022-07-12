#! /usr/bin/env bash

DEFAULT_CONFIG_FILE_NAME=test_sw.ini

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
    let t_end=${9}*${10}
    sed -i ${CONFIG_FILE} -e 's/^preconditioner *= *.*$/preconditioner = '${1}'/' \
                          -e 's/^nbsolpts *= *.*$/nbsolpts = '${2}'/' \
                          -e 's/^nb_elements_horizontal *= *.*$/nb_elements_horizontal = '${3}'/' \
                          -e 's/^precond_tolerance *= *.*$/precond_tolerance = '${4}'/' \
                          -e 's/^num_mg_levels *= *.*$/num_mg_levels = '${5}'/' \
                          -e 's/^mg_smoothe_only *= *.*$/mg_smoothe_only = '${6}'/' \
                          -e 's/^num_pre_smoothe *= *.*$/num_pre_smoothe = '${7}'/' \
                          -e 's/^num_post_smoothe *= *.*$/num_post_smoothe = '${8}'/' \
                          -e 's/^dt *= *.*$/dt = '${9}'/' \
                          -e 's/^t_end *= *.*$/t_end = '${t_end}'/' \
                          -e 's/^dg_to_fv_interp *= *.*$/dg_to_fv_interp = '${11}'/' \
                          -e 's/^linear_solver *= *.*$/linear_solver = '${12}'/' \
                          -e 's/^mg_smoother *= *.*$/mg_smoother = '${13}'/' \
                          -e 's/^kiops_dt_factor *= *.*$/kiops_dt_factor = '${14}'/'
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

    echo "${3} / ($order * $num_el) * $factor * 120" | bc | sed -e 's/\.[0-9]*//'
}

time_step=1800
num_steps=1
order=4
nb_elements=10

precond="fv-mg"
precond_tolerance=1e-2
num_mg_levels=4
mg_smoothe_only=1
num_pre_smoothe=1
num_post_smoothe=1

mg_smoother="kiops"
kiops_dt_factor=1.1

dg_to_fv_interp="lagrange"
# linear_solver="mg"
linear_solver="fgmres"


if [ "x${1}" == "x--gen-configs" ]; then

    ref_time_step=1800
    orders="4"
    # orders=2
    # element_counts="30 60 120"
    element_counts="60 120"
    # smoothings="0 1 2"
    smoothe_patterns="01 10 11 12 21 22 33"
    # smoothe_patterns="33"
    mg_levels="1 2 3 4"
    pre_tols="1e-1 1e-2 1e-5"
    preconds="p-mg fv-mg"
    smoothe_onlys="1 0"
    kiops_factors="0.8 1.0 1.2"
    #interps="lagrange l2-norm"
    interps="lagrange"
    for order in ${orders}; do
        for nb_elements in ${element_counts}; do

            echo "Time step: ${time_step}"

            precond="none"
            linear_solver="fgmres"
            set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
            cp -i ${CONFIG_FILE} "${CONFIG_DIR}/config.a${precond}_${order}_${nb_elements}_${time_step}_${num_steps}_${linear_solver}.ini"

            # linear_solver="multigrid"
            # num_mg_levels=${order}
            # num_pre_smoothe=1
            # num_post_smoothe=1
            # # for num_mg_levels in 1 2 4 8; do
            #     # for sm in 2 5 10; do
            #         # num_pre_smoothe=${sm}
            #         # num_post_smoothe=${sm}
            #         set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
            #         cp ${CONFIG_FILE} "${CONFIG_DIR}/config.b${precond}_${order}_${nb_elements}_${time_step}_${num_steps}_${linear_solver}_${num_mg_levels}_${mg_smoother}_${num_pre_smoothe}_${num_post_smoothe}.ini"
            #     # done
            # # done

            linear_solver="fgmres"
            precond="fv"
            for precond_tolerance in ${pre_tols}; do
                for dg_to_fv_interp in ${interps}; do
                    set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
                    cp -i ${CONFIG_FILE} "${CONFIG_DIR}/config.z${precond}_${order}_${nb_elements}_${dg_to_fv_interp:0:3}_${precond_tolerance}_${time_step}_${num_stept}.ini"
                done
            done

            precond="fv-mg"
            dg_to_fv_interp="lagrange"
            for precond in ${preconds}; do
                for num_mg_levels in ${mg_levels}; do
                    [ $num_mg_levels -gt $order ] && continue
                    for sm in ${smoothe_patterns}; do
                        num_pre_smoothe=${sm:0:1}
                        num_post_smoothe=${sm:1:1}
                        for kiops_dt_factor in ${kiops_factors}; do
                            mg_smoothe_only=1
                            precond_tolerance=0
                            set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
                            cp -i ${CONFIG_FILE} "${CONFIG_DIR}/config.c${precond}_${order}_${nb_elements}_${precond_tolerance}_${num_mg_levels}_${mg_smoother}_${mg_smoothe_only}_${num_pre_smoothe}_${num_post_smoothe}_${time_step}_${num_steps}_${dg_to_fv_interp:0:3}_${kiops_dt_factor}.ini"

                            mg_smoothe_only=0
                            for precond_tolerance in ${pre_tols}; do
                                set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
                                cp -i ${CONFIG_FILE} "${CONFIG_DIR}/config.c${precond}_${order}_${nb_elements}_${precond_tolerance}_${num_mg_levels}_${mg_smoother}_${mg_smoothe_only}_${num_pre_smoothe}_${num_post_smoothe}_${time_step}_${num_steps}_${dg_to_fv_interp:0:3}_${kiops_dt_factor}.ini"
                            done
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
    set_parameters ${precond} ${order} ${nb_elements} ${precond_tolerance} ${num_mg_levels} ${mg_smoothe_only} ${num_pre_smoothe} ${num_post_smoothe} ${time_step} ${num_steps} ${dg_to_fv_interp} ${linear_solver} ${mg_smoother} ${kiops_dt_factor}
    if [ "x${1}" != "x--just-config" ]; then
        run_program
    fi
fi
