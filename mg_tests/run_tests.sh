#! /usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=${SCRIPT_DIR}/..
GEF_MAIN=main_gef.py
CONFIG_FILE=${SCRIPT_DIR}/test.ini
CONFIG_DIR=${SCRIPT_DIR}/configs

MPIRUN="aprun"
# MPIRUN="mpirun -tag-output"

cd ${SCRIPT_DIR}
mkdir -p ${CONFIG_DIR}

function run_program() {
    if [ $# -gt 0 ]; then
        the_config=${1}
    else
        the_config=${CONFIG_FILE}
    fi
    set -x
    ${MPIRUN} -n 6 python3 ${GEF_DIR}/${GEF_MAIN} ${the_config} 2>&1 | grep ',0]'
    set +x
}

function set_parameters() {
    sed -i ${CONFIG_FILE} -e 's/^use_preconditioner *= *[0-9][0-9]*$/use_preconditioner = '${1}'/' \
                          -e 's/^nbsolpts *= *[0-9][0-9]*$/nbsolpts = '${2}'/' \
                          -e 's/^nb_elements_horizontal *= *[0-9][0-9]*$/nb_elements_horizontal = '${3}'/' \
                          -e 's/^precond_tolerance *= *.*$/precond_tolerance = '${4}'/' \
                          -e 's/^max_mg_level *= *.*$/max_mg_level = '${5}'/' \
                          -e 's/^mg_smoothe_only *= *.*$/mg_smoothe_only = '${6}'/' \
                          -e 's/^num_pre_smoothing *= *.*$/num_pre_smoothing = '${7}'/' \
                          -e 's/^num_post_smoothing *= *.*$/num_post_smoothing = '${8}'/' \
                          -e 's/^mg_cfl *= *.*$/mg_cfl = '${9}'/' \
                          -e 's/^dt *= *.*$/dt = '${10}'/' \
                          -e 's/^t_end *= *.*$/t_end = '${10}'/' \
                          -e 's/^dg_to_fv_interp *= *.*$/dg_to_fv_interp = '${11}'/' \
                          -e 's/^linear_solver *= *.*$/linear_solver = '${12}'/'
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

time_step=7200
use_precond=2
order=2
nb_elements=30
precond_tolerance=1e-1
max_mg_level=0
mg_smoothe_only=1
num_pre_smoothing=1
num_post_smoothing=1
# mg_cfl=0.9
mg_cfl=3.0
dg_to_fv_interp="lagrange"
linear_solver="fgmres"


if [ "x${1}" == "x--gen-configs" ]; then

    ref_time_step=7200
    orders="2 4 8"
    # orders=2
    # element_counts="30 60 120"
    element_counts="30 60 120"
    smoothings="1 2 3"
    # smoothings="4"
    mg_levels="0 1 2 3"
    pre_tols="1e-1 1e-7"
    #interps="lagrange l2-norm"
    interps="lagrange"
    for order in ${orders}; do
        for nb_elements in ${element_counts}; do
            [ $nb_elements -gt 60 ] && [ $order -gt 2 ] && continue
            [ $nb_elements -gt 30 ] && [ $order -gt 4 ] && continue

            time_step=$(compute_time_step ${order} ${nb_elements} ${ref_time_step})

            echo "Time step: ${time_step}"

            use_precond=0
            linear_solver="fgmres"
            set_parameters ${use_precond} ${order} ${nb_elements} ${precond_tolerance} ${max_mg_level} ${mg_smoothe_only} ${num_pre_smoothing} ${num_post_smoothing} ${mg_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver}
            cp ${CONFIG_FILE} "${CONFIG_DIR}/config.a${use_precond}_${order}_${nb_elements}_${time_step}_${linear_solver}.ini"

            linear_solver="multigrid"
            max_mg_level=0
            num_pre_smoothing=1
            num_post_smoothing=1
            # for max_mg_level in 0 1; do
                # for sm in 2 5 10; do
                    # num_pre_smoothing=${sm}
                    # num_post_smoothing=${sm}
                    set_parameters ${use_precond} ${order} ${nb_elements} ${precond_tolerance} ${max_mg_level} ${mg_smoothe_only} ${num_pre_smoothing} ${num_post_smoothing} ${mg_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver}
                    cp ${CONFIG_FILE} "${CONFIG_DIR}/config.b${use_precond}_${order}_${nb_elements}_${time_step}_${linear_solver}_${max_mg_level}_${num_pre_smoothing}_${num_post_smoothing}.ini"
                # done
            # done

            linear_solver="fgmres"
            use_precond=1
            for precond_tolerance in ${pre_tols}; do
                for dg_to_fv_interp in ${interps}; do
                    set_parameters ${use_precond} ${order} ${nb_elements} ${precond_tolerance} ${max_mg_level} ${mg_smoothe_only} ${num_pre_smoothing} ${num_post_smoothing} ${mg_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver}
                    cp ${CONFIG_FILE} "${CONFIG_DIR}/config.z${use_precond}_${order}_${nb_elements}_${dg_to_fv_interp:0:3}_${precond_tolerance}_${time_step}.ini"
                done
            done

            use_precond=2
            dg_to_fv_interp="lagrange"
            for max_mg_level in ${mg_levels}; do
                [ $max_mg_level -gt 1 ] && [ $order -lt 4 ] && continue
                [ $max_mg_level -gt 2 ] && [ $order -lt 8 ] && continue
                for mg_smoothe_only in 1; do
                    for num_pre_smoothing in ${smoothings}; do
                        num_post_smoothing=${num_pre_smoothing}
                        set_parameters ${use_precond} ${order} ${nb_elements} ${precond_tolerance} ${max_mg_level} ${mg_smoothe_only} ${num_pre_smoothing} ${num_post_smoothing} ${mg_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver}
                        cp ${CONFIG_FILE} "${CONFIG_DIR}/config.c${use_precond}_${order}_${nb_elements}_${precond_tolerance}_${max_mg_level}_${mg_smoothe_only}_${num_pre_smoothing}_${num_post_smoothing}_${mg_cfl}_${time_step}_${dg_to_fv_interp:0:3}.ini"
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
    set_parameters ${use_precond} ${order} ${nb_elements} ${precond_tolerance} ${max_mg_level} ${mg_smoothe_only} ${num_pre_smoothing} ${num_post_smoothing} ${mg_cfl} ${time_step} ${dg_to_fv_interp} ${linear_solver}
    run_program
fi
