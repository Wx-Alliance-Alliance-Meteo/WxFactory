#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=${SCRIPT_DIR}/..
TEST_DIR="${GEF_DIR}/testing_gef.tmp"
LISTING=${TEST_DIR}/listing.txt

CART_CONFIG_ORIG=${GEF_DIR}/config/gaussian_bubble.ini
CART_CONFIG_BASE=${TEST_DIR}/gaussian_bubble.ini

CUBE_CONFIG_ORIG=${GEF_DIR}/config/dcmip31.ini
CUBE_CONFIG_BASE=${TEST_DIR}/dcmip31.ini

mkdir -pv ${TEST_DIR}
rm -rf ${TEST_DIR}/*


function set_param() {
    file=${1}
    shift 1
    for arg in $@; do
        # echo "${arg} = ${!arg}"
        # sed_str='s/^'${arg}'\s*=\s*.*/'${arg}' = '${!arg}'/'
        # echo "sed str: ${sed_str}"
        sed -i ${file} -e 's|^'${arg}'\s*=\s*.*|'${arg}' = '${!arg}'|'
        grep -q "^${arg}\s*=" ${file} || echo "${arg} = ${!arg}" >> ${file}
    done
}

function test_cart2d() {
    dt=5
    t_end=5
    time_integrator=ros2
    tolerance=1e-6
    starting_step=0
    gmres_restart=30
    nbsolpts=4
    nb_elements_horizontal=5
    nb_elements_vertical=8
    preconditioner=none
    output_freq=1
    save_state_freq=1
    store_solver_stats=1
    output_dir=${TEST_DIR}

    cp ${CART_CONFIG_ORIG} ${CART_CONFIG_BASE}
    set_param ${CART_CONFIG_BASE} dt t_end time_integrator tolerance starting_step gmres_restart        \
                                nbsolpts nb_elements_horizontal nb_elements_vertical preconditioner   \
                                output_freq save_state_freq store_solver_stats output_dir


    time_integrators="epi2 epi3 epi_stiff3 epi_stiff4 srerk3 imex2 tvdrk3 ros2 rosexp2 partrosexp2 strang_epi2_ros2 strang_ros2_epi2"
    for time_integrator in ${time_integrators}; do
        # Set up params
        cfg=${CART_CONFIG_BASE}.${time_integrator}
        cp ${CART_CONFIG_BASE} ${cfg}
        set_param ${cfg} time_integrator
        echo "Testing time integrator (2D Euler): ${time_integrator}"

        # Run test
        ${GEF_DIR}/main_gef.py ${cfg} > ${LISTING} 2>&1 && continue

        # In case of error, stop
        echo "AHHHHHhhhh error with time integrator ${time_integrator}"
        cat ${LISTING}
        return 1
    done

    # All tests were successful
    return 0
}

function test_cube_sphere() {
    dt=30
    t_end=30
    time_integrator=ros2
    tolerance=1e-7
    starting_step=0
    gmres_restart=200
    nbsolpts=4
    nb_elements_horizontal=4
    nb_elements_vertical=8
    preconditioner=none
    output_freq=1
    save_state_freq=1
    store_solver_stats=1
    output_dir=${TEST_DIR}

    cp ${CUBE_CONFIG_ORIG} ${CUBE_CONFIG_BASE}
    set_param ${CUBE_CONFIG_BASE} dt t_end time_integrator tolerance starting_step gmres_restart        \
                                nbsolpts nb_elements_horizontal nb_elements_vertical preconditioner   \
                                output_freq save_state_freq store_solver_stats output_dir

    time_integrators="epi2 epi3 epi_stiff3 epi_stiff4 srerk3 tvdrk3 ros2"
    no_time_integrators="imex2 rosexp2 partrosexp2 strang_epi2_ros2 strang_ros2_epi2"
    for time_integrator in ${time_integrators}; do
        # Set up params
        cfg=${CUBE_CONFIG_BASE}.${time_integrator}
        cp ${CUBE_CONFIG_BASE} ${cfg}
        set_param ${cfg} time_integrator
        echo "Testing time integrator (3D Euler): ${time_integrator}"

        # Run test
        mpirun -n 6 ${GEF_DIR}/main_gef.py ${cfg} > ${LISTING} 2>&1 && continue

        # In case of error, stop
        echo "AHHHHHhhhh error with time integrator ${time_integrator}"
        cat ${LISTING}
        return 1
    done

    # All tests were successful
    return 0
}

test_cart2d && 
test_cube_sphere && 
echo "Test successful!" && rm -rf ${TEST_DIR}
