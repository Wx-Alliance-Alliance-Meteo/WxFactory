#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTPUT_DIR=/home/vma000/site5/wx_factory/cpu_scaling
BASE_WORK_DIR=${SCRIPT_DIR}/tmp
WORK_DIR=/dev/null
WX_DIR=${SCRIPT_DIR}/../..
BASE_JOB=${SCRIPT_DIR}/base.job

mkdir -pv ${OUTPUT_DIR}
mkdir -pv ${BASE_WORK_DIR}

function gen_config() {
    dt=30
    t_end=90
    starting_step=2
    num_solpts=$1
    num_elem_hori=$2
    num_elem_vert=$3
    device=$4
    cat << EOF
        [General]
        equations = Euler
        depth_approx = shallow

        [System]
        desired_device = ${device}

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
        time_integrator = epi2
        tolerance = 1e-7
        exponential_solver = kiops
        starting_step = $starting_step

        [Spatial_discretization]
        num_solpts = $num_solpts
        num_elements_horizontal = $num_elem_hori
        num_elements_vertical = $num_elem_vert

        [Output_options]
        save_state_freq = 0 
        store_solver_stats = 1
        output_dir = ${OUTPUT_DIR}
        solver_stats_file = cpu_scaling.db
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

WTIME=40
WTIME_GPU=20
NODE_SIZE=80
device=cpp

mkdir -pv ${WORK_DIR}/$device || exit -1
gen_config 4 72 12 $device > ${WORK_DIR}/$device/config.ini

# cpu_counts="6 24 54 96 216 384 486 864 1944 3456 7776"
cpu_counts="24 54 96"
for num_cpus in ${cpu_counts}; do
for config in ${WORK_DIR}/$device/*.ini; do
    c=$(basename ${config})
    echo "Config $c"

    printf -v num_cpus_str "%05d" ${num_cpus}

    JOB_SCRIPT=${WORK_DIR}/${device}/${c}_${num_cpus_str}.job
    sed -e 's|^WX_DIR=.*|WX_DIR='${WX_DIR}'|' \
        -e 's|^CONFIG_FILE=.*|CONFIG_FILE='${config}'|' \
        -e 's|^NUM_CPUS=.*|NUM_CPUS='${num_cpus}'|' \
        -e 's|^\(#PBS -N \).*|\1scaling_'${num_cpus_str}'|'\
        < ${BASE_JOB} > ${JOB_SCRIPT}  || exit -1

    if [ ${num_cpus} -gt ${NODE_SIZE} ]; then
        num_nodes=$(((${num_cpus} + ${NODE_SIZE} - 1) / ${NODE_SIZE}))
        sed -e 's|^\(#PBS -l .*\)select=[0-9][0-9]*|\1select='${num_nodes}'|' -i ${JOB_SCRIPT}
    fi

    qsub ${JOB_SCRIPT}
  
done # config
done # cpu_count
