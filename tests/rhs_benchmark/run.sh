#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTPUT_DIR=/home/vma000/site5/wx_factory/rhs_benchmark
BASE_WORK_DIR=${SCRIPT_DIR}/tmp
WORK_DIR=/dev/null
WX_DIR=${SCRIPT_DIR}/../..

mkdir -pv ${OUTPUT_DIR}
mkdir -pv ${BASE_WORK_DIR}

function gen_config() {
    dt=30
    t_end=360
    starting_step=10
    num_solpts=$1
    num_elem_hori=$2
    num_elem_vert=$3
    cat << EOF
        [General]
        equations = Euler
        depth_approx = shallow

        [System]
        desired_device = cpu

        [Grid]
        grid_type = cubed_sphere
        lambda0 = 0.0
        phi0 = 0.7853981633974483
        alpha0 = 0.0
        ztop = 10000.0

        [Test_case]
        case_number = 31

        [Time_integration]
        dt = $dt  
        t_end = $t_end 
        starting_step = ${starting_step}
        time_integrator = epi2
        tolerance = 1e-7
        exponential_solver = kiops

        [Spatial_discretization]
        num_solpts = $num_solpts
        num_elements_horizontal = $num_elem_hori
        num_elements_vertical = $num_elem_vert

        [Output_options]
        save_state_freq = 10 
        store_solver_stats = 1
        output_dir = ${OUTPUT_DIR}
        solver_stats_file = rhs_benchmark.db
EOF
}

JOB_NAME="no_name"
for i in {0..1000}; do
    d=$(date +%Y%m%d_%H%M%S)
    JOB_NAME=rhs_benchmark.$d.$i
    WORK_DIR=${BASE_WORK_DIR}/$d.$i
    test -d ${WORK_DIR} || break
done
mkdir -pv ${WORK_DIR} || exit -1

gen_config 2 30 30 > ${WORK_DIR}/c1.ini
gen_config 3 20 20 > ${WORK_DIR}/c2.ini
gen_config 4 15 15 > ${WORK_DIR}/c3.ini
gen_config 5 12 12 > ${WORK_DIR}/c4.ini
gen_config 6 10 10 > ${WORK_DIR}/c5.ini

WTIME=180
WTIME_GPU=20
for config in ${WORK_DIR}/*; do
    c=$(basename ${config})
    echo "Config $c"

    JOB_SCRIPT=${WORK_DIR}/${c}.job
    sed -e 's|^WX_DIR=.*|WX_DIR='${WX_DIR}'|' \
        -e 's|^CONFIG_FILE=.*|CONFIG_FILE='${config}'|' \
        < base.job > ${JOB_SCRIPT}

    LAUNCH_COMMAND_CPU="ord_soumet -cpus 80 -w ${WTIME} -jn ${JOB_NAME} -mpi -jobfile ${JOB_SCRIPT} -listing ${WORK_DIR} -cm 2000M -waste 100"
    LAUNCH_COMMAND_GPU="ord_soumet -cpus 1x12 -gpus 1 -w ${WTIME} -jn ${JOB_NAME} -mpi -jobfile ${JOB_SCRIPT} -listing ${WORK_DIR} -cm 6000M -waste 100 -mach underhill"
    # LAUNCH_COMMAND=${LAUNCH_COMMAND_CPU}
    LAUNCH_COMMAND=${LAUNCH_COMMAND_GPU}
    echo ${LAUNCH_COMMAND}
    ${LAUNCH_COMMAND}
done
