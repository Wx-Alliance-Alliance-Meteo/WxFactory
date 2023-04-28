#!/bin/bash

# Siqi Wei
# Jan 2023 
# Script to run the bulle problem using operator splitting 


# Save the current directory location
startingDir=`pwd`

# Bubble location
bubbleDir=~/Documents/ECCC/gef_pull_april17/gef

# Config file location 
configDir=$bubbleDir/config

# Time integrator 
# timeIntegrator=strang_epi2_ros2

# Testoutput location and file
testoutputDir=$bubbleDir/vicky/testoutput 

# config file name 
configFile=gaussian_bubble_testOS.ini 

# testing stepsizes 
stepsize="2.0" # 1.0 0.5 0.25 0.1"
#numofsteps="300.0 350.0 400.0 450.0"

# OS22 parameter
OS22param="0.0 0.1 0.2 0.3 0.4 0.5"

tf=500

tol=1e-10

for dt in $stepsize
do

    echo $dt
    #for os22b in $OS22param
    #do
    #    echo $os22b
        timeIntegrator=sdirk
        #strang_epi6_ex_epi6_im
        #os22_ros2_epi2[$os22b]
        testoutputFile=${timeIntegrator}_tol_${tol}_dt_${dt}_tf_${tf}
        echo $timeIntegrator
        echo -e "
[General]
equations = euler

[Grid]
grid_type = cartesian2d

x0 = 0
x1 = 1000
z0 = 0
z1 = 1500

[Test_case]

# Possible values
#   1  = constant
#   2  = gaussian
#   3  = cold bubble
case_number = 2

bubble_theta = 303.15
bubble_rad = 250.


[Time_integration]
# Time step
dt = $dt

# End time of the simulation in sec
t_end = $tf

# Time integration scheme
# Possible values  = 'epi2', 'epi3', 'epi4', 'epi5', 'epi6'  = multistep exponential propagation iterative
#                    'epi_stiff[3-6]' = stiffness resilient multistep exponential propagation iterative
#                    'srerk[3-6]' = stiffness resilient exponential Runge-Kutta
#                    'tvdrk3' = 3rd order TVD Runge-Kutta
#                    'ros2' = 2nd order semi-implicit
#                    'imex2' = 2nd order IMEX
#                    'strang_epi2_ros2' = Strang splitting with Epi2 first and Rat2 second
#                    'strang_ros2_epi2' = Strang splitting with Rat2 first and Epi2 second
#                    'rosexp2' = Implicit-Exponential hybrid scheme (exponential first)
#                    'partrosexp2' = Implicit-Exponential hybrid scheme (implicit first)
#                    'crank_nicolson' = 2nd order implicit scheme
#                    'bdf2' = 2nd order implicit multistep
time_integrator = $timeIntegrator

# Initial size of the Krylov space. Stay constant for phipm, but can be updated dynamically with phipm_iom
krylov_size = 10

# Solver tolerance
tolerance = $tol

jacobian_method = FD

# If you want to start at an arbitrary timestep (assuming its state vector is already saved in a file)
starting_step = 0

verbose_solver = 1
gmres_restart = 30

[Preconditioning]
preconditioner = none
precond_tolerance = 1e-1
num_mg_levels = 5
num_pre_smoothe = 1
num_post_smoothe = 1
pseudo_cfl = 5.0
restrict_method = modal
kiops_dt_factor = 1.2
mg_solve_coarsest = 0
mg_smoother = exp
precond_filter_apply = 0
verbose_precond = 0


[Spatial_discretization]

# The grid will have (nbsolpts) x (nbsolpts) nodal points in each elements.
nbsolpts = 5

# Number of element in x^1 and x^2 directions
# Each face of the cube have (nbElements x nbElements) elements, for a total of (6 x nbElements x nbElements) elements.
nb_elements_horizontal = 19
nb_elements_vertical   = 30

# Weak filter
filter_apply = 0
filter_order = 4
filter_cutoff = 0.5

[Output_options]

# Print blockstats every "stat_freq" steps, 0 to disable.
stat_freq = 1

# Plot solution every "output_freq" steps, 0 to disable.
output_freq = 0

# Save the state vector to a file at every "save_state_freq" steps. 0 to disable.
save_state_freq = 2000

# Store statistics about the solver (iterations, residuals, etc.). 0 to disable.
store_solver_stats = 1

# Set graph and state vector Output directory
output_dir = ${testoutputDir}/${testoutputFile}


        " > $configDir/$configFile

        python ${bubbleDir}/main_gef.py ./config/${configFile} > ${testoutputDir}/${testoutputFile}.txt

#done
done





