#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=${SCRIPT_DIR}/..
ORIG_CONFIG=${GEF_DIR}/config/dcmip31.ini
# ORIG_CONFIG=${GEF_DIR}/config/gaussian_bubble.ini
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
nb_elements_horizontal=80
nb_elements_vertical=8
RESULT_DIR=${GEF_DIR}/multi_config_results
# RESULT_DIR=${GEF_DIR}/multi_config_results

dt=30
t_end=180
time_integrator=epi2
tolerance=1e-7
gmres_restart=100
nbsolpts=4
preconditioner=none
output_freq=0
save_state_freq=0
store_solver_stats=1
output_dir=${RESULT_DIR}
solver_stats_file=solver_stats_7.db
starting_step=0

cp ${ORIG_CONFIG} ${TMP_BASE_CONFIG}
set_param_strict ${TMP_BASE_CONFIG} dt t_end time_integrator tolerance starting_step gmres_restart          \
                             nbsolpts nb_elements_horizontal nb_elements_vertical preconditioner            \
                             output_freq save_state_freq store_solver_stats output_dir solver_stats_file

do_no_precond=0
do_fv_precond=0
do_erk1_smoother=0
do_erk3_smoother=0
do_exp_smoother=0
do_ark3_smoother=1

# for tolerance in 1e-6 1e-7; do
for tolerance in 1e-6; do
    [ ${tolerance} == 1e-6 ] && solver_stats_file=solver_stats_6.db
    [ ${tolerance} == 1e-7 ] && solver_stats_file=solver_stats_7.db

    # No precond
    if [ ${do_no_precond} -gt 0 ]; then
        time_integrator=epi2
        make_config ${DEST_DIR}/epi2_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}.ini \
            time_integrator tolerance solver_stats_file

        time_integrator=ros2
        make_config ${DEST_DIR}/ros2_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}.ini \
            time_integrator tolerance solver_stats_file
    fi

    # FV precond
    if [ ${do_fv_precond} -gt 0 ]; then
        time_integrator=ros2
        preconditioner=fv
        for precond_tolerance in 6e-1 5e-1 4e-1 3e-1 2e-1 1e-1 5e-2 2e-2; do
            make_config ${DEST_DIR}/fv_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}_${precond_tolerance}.ini \
                time_integrator precond_tolerance preconditioner tolerance solver_stats_file
        done
    fi


    # FV-MG precond
    if [ ${do_erk3_smoother} -gt 0 ]; then
        time_integrator=ros2
        preconditioner=fv-mg
        mg_smoother=erk3
        mg_solve_coarsest=0
        for num_pre_smoothe in 0 1 2; do
        for num_post_smoothe in 0 1 2; do
        [ $(($num_pre_smoothe + $num_post_smoothe)) -lt 1 ] && continue
        [ $(($num_pre_smoothe + $num_post_smoothe)) -gt 3 ] && continue
        # for pseudo_cfl in 16.5e4 17.0e4 17.5e4 18.0e4 18.5e4 19.0e4 19.5e4 20.0e4; do
        for pseudo_cfl in $(seq 12 2 16); do
        for precond_tolerance in 1e-1; do
            make_config ${DEST_DIR}/fv-mg_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}_${mg_smoother}_${num_pre_smoothe}${num_post_smoothe}${mg_solve_coarsest}_${pseudo_cfl}_${precond_tolerance}.ini \
                time_integrator preconditioner mg_smoother num_pre_smoothe num_post_smoothe pseudo_cfl precond_tolerance tolerance solver_stats_file mg_solve_coarsest
        done
        done
        done
        done
    fi

    if [ ${do_ark3_smoother} -gt 0 ]; then
        time_integrator=ros2
        preconditioner=fv-mg
        mg_smoother=ark3
        mg_solve_coarsest=0
        gmres_restart=100
        for num_pre_smoothe in 0 1 2; do
        for num_post_smoothe in 0 1 2; do
        [ $(($num_pre_smoothe + $num_post_smoothe)) -lt 1 ] && continue
        [ $(($num_pre_smoothe + $num_post_smoothe)) -gt 3 ] && continue
        # for pseudo_cfl in 16.5e4 17.0e4 17.5e4 18.0e4 18.5e4 19.0e4 19.5e4 20.0e4; do
        for pseudo_cfl in $(seq 0.65 0.05 0.85); do
        for precond_tolerance in 1e-1; do
            make_config ${DEST_DIR}/fv-mg_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}_${mg_smoother}_${num_pre_smoothe}${num_post_smoothe}${mg_solve_coarsest}_${pseudo_cfl}_${precond_tolerance}.ini \
                time_integrator preconditioner mg_smoother num_pre_smoothe num_post_smoothe pseudo_cfl precond_tolerance tolerance solver_stats_file mg_solve_coarsest gmres_restart
        done
        done
        done
        done
    fi

    if [ ${do_erk1_smoother} -gt 0 ]; then
        time_integrator=ros2
        preconditioner=fv-mg
        mg_smoother=erk1
        mg_solve_coarsest=0
        for num_pre_smoothe in 1 2 3 4; do
        for num_post_smoothe in 1 2 3 4; do
        [ $(($num_pre_smoothe + $num_post_smoothe)) -lt 2 ] && continue
        [ $(($num_pre_smoothe + $num_post_smoothe)) -gt 5 ] && continue
        # for pseudo_cfl in 4.0e4 4.2e4 4.4e4 4.6e4 4.8e4 5.0e4; do
        # for pseudo_cfl in 7.4e4 7.6e4; do
        for pseudo_cfl in $(seq 4.0 0.5 9.0); do
        for precond_tolerance in 1e-1; do
            make_config ${DEST_DIR}/fv-mg_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}_${mg_smoother}_${num_pre_smoothe}${num_post_smoothe}${mg_solve_coarsest}_${pseudo_cfl}_${precond_tolerance}.ini \
                time_integrator preconditioner mg_smoother num_pre_smoothe num_post_smoothe pseudo_cfl precond_tolerance tolerance solver_stats_file mg_solve_coarsest
        done
        done
        done
        done
    fi

    if [ ${do_exp_smoother} -gt 0 ]; then
        time_integrator=ros2
        preconditioner=fv-mg
        mg_smoother=exp
        precond_tolerance=1e-1
        for num_pre_smoothe in 0 1 2; do
        for num_post_smoothe in 0 1 2; do
        [ $(($num_pre_smoothe + $num_post_smoothe)) -lt 1 ] && continue
        for lvl0 in 1.9 2.0 2.1; do
        for lvl1 in 2.1 2.3 2.5; do
        for lvl2 in 1.5 1.7 1.9 2.1; do
            mg_solve_coarsest=1
            exp_smoothe_spectral_radii="[${lvl0}, ${lvl1}, ${lvl2}]"
            for precond_tolerance in 2.0 1.5 1.0 9e-1; do
                make_config ${DEST_DIR}/fv-mg_${nb_elements_horizontal}x${nb_elements_vertical}_${tolerance}_${mg_smoother}_${lvl0}-${lvl1}-${lvl2}-${num_pre_smoothe}${num_post_smoothe}${mg_solve_coarsest}_${precond_tolerance}.ini \
                    time_integrator preconditioner mg_smoother exp_smoothe_spectral_radii precond_tolerance tolerance \
                    solver_stats_file mg_solve_coarsest num_pre_smoothe num_post_smoothe
            done
        done
        done
        done
        done
        done
    fi
done # Tolerances

echo "Generated ${num_configs} configs"
actual_num_configs=$(ls ${DEST_DIR} | wc -w)
[ ${actual_num_configs} -eq ${num_configs} ] || echo "There should be $num_configs configs. There are $actual_num_configs."

rm ${TMP_BASE_CONFIG}
