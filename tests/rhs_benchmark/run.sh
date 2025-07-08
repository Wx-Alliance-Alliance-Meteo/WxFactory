#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTPUT_DIR=/home/vma000/site5/wx_factory/rhs_benchmark
BASE_WORK_DIR=${SCRIPT_DIR}/tmp
WORK_DIR=/dev/null
WX_DIR=${SCRIPT_DIR}/../..
BASE_JOB=${SCRIPT_DIR}/base.job

mkdir -pv ${OUTPUT_DIR}
mkdir -pv ${BASE_WORK_DIR}

function gen_config() {
    dt=30
    t_end=360
    starting_step=10
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

WTIME=180
WTIME_GPU=20
for device in cpp cuda numpy cupy; do
# for device in cupy cuda; do
    mkdir -pv ${WORK_DIR}/$device || exit -1
    gen_config 2 30 30 $device > ${WORK_DIR}/$device/c01.ini
    gen_config 3 20 20 $device > ${WORK_DIR}/$device/c02.ini
    gen_config 4 15 15 $device > ${WORK_DIR}/$device/c03.ini
    gen_config 5 12 12 $device > ${WORK_DIR}/$device/c04.ini
    gen_config 6 10 10 $device > ${WORK_DIR}/$device/c05.ini

    # gen_config 2 60 30 $device > ${WORK_DIR}/$device/c06.ini
    # gen_config 3 40 20 $device > ${WORK_DIR}/$device/c07.ini
    # gen_config 4 30 15 $device > ${WORK_DIR}/$device/c08.ini
    # gen_config 5 24 12 $device > ${WORK_DIR}/$device/c09.ini
    # gen_config 6 20 10 $device > ${WORK_DIR}/$device/c10.ini

    for config in ${WORK_DIR}/$device/*; do
        c=$(basename ${config})
        echo "Config $c"

        JOB_SCRIPT=${WORK_DIR}/${device}/${c}.job
        sed -e 's|^WX_DIR=.*|WX_DIR='${WX_DIR}'|' \
            -e 's|^CONFIG_FILE=.*|CONFIG_FILE='${config}'|' \
            < ${BASE_JOB} > ${JOB_SCRIPT}  || exit -1

        if [ "cpp" == ${device} ] || [ "numpy" == "${device}" ]; then
            sed -i ${JOB_SCRIPT} \
                -e 's|^#PBS -l select=.*|#PBS -l select=1:ncpus=80:mpiprocs=80:mem=180gb|' \
                -e 's|^#SBATCH --partition=.*|#SBATCH --partition=standard|' \
                -e 's|^#SBATCH --account=.*|#SBATCH --account=eccc_mrd|'
        elif [ "cuda" == "${device}" ] || [ "cupy" == "${device}" ]; then
            sed -i ${JOB_SCRIPT} \
                -e 's|^#PBS -l select=.*|#PBS -l select=2:ncpus=48:mpiprocs=4:ngpus=4:mem=205gb\n#PBS -q gpu|' \
                -e 's|^#SBATCH --partition=.*|#SBATCH --partition=gpu_a100|' \
                -e 's|^#SBATCH --account=.*|#SBATCH --account=eccc_mrd__gpu_a100\n#SBATCH --gpus=1\n|'
        else
            echo "Whooah something wrong"
            exit -1
        fi

        if which sbatch 2>/dev/null ; then
            LAUNCH_COMMAND="sbatch --ignore-pbs ${JOB_SCRIPT}"
        else
            LAUNCH_COMMAND="qsub ${JOB_SCRIPT}"
        fi

        echo ${LAUNCH_COMMAND}
        ${LAUNCH_COMMAND}
    done
done
