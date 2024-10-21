#!/usr/bin/env bash

# set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=${SCRIPT_DIR}/..
# ORIG_CONFIG=${GEF_DIR}/config/dcmip31.ini
TMP_BASE_CONFIG=${ORIG_CONFIG}.tmp
DEST_DIR=${GEF_DIR}/test_configs

mkdir -pv ${DEST_DIR}
rm -rf ${DEST_DIR}/*

function set_param() {
    file=${1}
    shift 1
    sed_script=param_script.sed
    rm -f $sed_script
    for arg in $@; do
        echo 's|^'${arg}'\s*=\s*.*|'${arg}' = '${!arg}'|' >> $sed_script
    done
    sed -i ${file} -f ${sed_script}
    rm -f $sed_script
}

function set_param_strict() {
    set_param $@
    file=${1}
    shift 1
    for arg in $@; do
        grep -q "^${arg}\s*=" ${file} || echo "Missing argument ${arg} in config file!!!"
    done
}

num_configs=0
function make_config() {
    cp ${TMP_BASE_CONFIG} ${1}
    set_param $@
    num_configs=$(($num_configs + 1))
}

# Problem size

# Choose common problem parameters here
dt=900
t_end=14400
time_integrator=epi6
tolerance=1e-12
nbsolpts=7
output_freq=0
exponential_solver=kiops
nb_elements_horizontal=84

RESULT_DIR=${GEF_DIR}/results_tanya_${nb_elements_horizontal}
output_dir=${RESULT_DIR}

# Choose your base configs here
base_configs="${GEF_DIR}/config/procs294/case5.ini ${GEF_DIR}/config/procs294/case6.ini ${GEF_DIR}/config/procs294/galewsky.ini"

for orig_config in ${base_configs}; do
    cp ${orig_config} ${TMP_BASE_CONFIG}
    case=$(expr match "$(basename ${orig_config})" '\(^[a-z][a-z]*[0-9]*\)')

    # Set common problem parameters
    set_param_strict ${TMP_BASE_CONFIG} dt t_end time_integrator tolerance nbsolpts nb_elements_horizontal output_freq output_dir exponential_solver
    
    ####### Add exponential solvers here ########
    exponential_solvers="kiops kiops_ne cwy_1s cwy_ne cwy_ne1s pmex pmex_1s pmex_ne pmex_ne1s"

    for exponential_solver in ${exponential_solvers}; do
        make_config ${DEST_DIR}/${case}_${exponential_solver}_${nb_elements_horizontal}.ini exponential_solver
    done
done

echo "Generated ${num_configs} configs"
actual_num_configs=$(ls ${DEST_DIR} | wc -w)
[ ${actual_num_configs} -eq ${num_configs} ] || echo "There should be $num_configs configs. There are $actual_num_configs."

rm ${TMP_BASE_CONFIG}
