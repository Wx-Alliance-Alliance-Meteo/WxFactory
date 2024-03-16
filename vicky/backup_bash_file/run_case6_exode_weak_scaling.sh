#!/bin/bash


#SBATCH --account=eccc_pegasus_mrd
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=64
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=vicky/EXODE_testoutput_case6/run_scripts/case6_exode_weak_scaling_%j.out
#SBATCH --error=vicky/EXODE_testoutput_case6/run_scripts/case6_exode_weak_scaling_%j.err


# Siqi Wei
# July 2023
# Script to run the case 6 to text EXODE

# Save the current directory location
startingDir=`pwd`

# Bubble location
GEFDir=/home/siw001/gef

# Config file location
configDir=$GEFDir/config


# Testoutput location and file
testoutputDir=$GEFDir/vicky/EXODE_testoutput_case6/weak_scaling
mkdir $testoutputDir

# config file name
configFile=case6_exode.ini

# testing stepsizes
stepsize="900" # 1.0 0.5 0.25 0.1"
#numofsteps="300.0 350.0 400.0 450.0"
#outputfreq=1152

#spacial_dimension
nbsolpts=4
nb_elements_sel="48 96 192 384 768" # 336" # 672" # 168 336" # 168 336"

tf=9000

tol=1e-7

# EXODE methods 
exodemethod="RK23 RK45 MERSON4 ARK3(2)4L[2]SA-ERK ERK3(2)3L ERK4(3)4L"


for dt in $stepsize
do

    #echo $dt
    for nb_elements in $nb_elements_sel  
    do
	
	if [ $nb_elements -eq 48 ] 
	then 
		core=6
	elif [ $nb_elements -eq 96 ]
	then 
	 	core=24
	elif [ $nb_elements -eq 192 ]
        then
                core=96
	elif [ $nb_elements -eq 384 ]
        then
                core=384
	elif [ $nb_elements -eq 768 ]
        then
                core=1536
	else
		echo check problem size
	fi
	
	echo nb_elements = ${nb_elements}
	echo core = $core



	for exode_method in $exodemethod
	do
		#echo $nb_elements

        	echo -e "
[General]

equations = shallow_water

[Grid]

# Possible values: cubed_sphere, cartesian2d
grid_type = cubed_sphere
λ0 = 0.0
#ϕ0 = 0.0
ϕ0 = 0.7853981633974483
α0 = 0.0

[Test_case]

# Possible values
#  -1 : multiscale signal (passive advection only)
#   0 : deformation flow  (passive advection only)
#   1 : cosine hill (passive advection only)
#   2 : zonal flow (shallow water)
#   5 : zonal flow over an isolated mountain (shallow water)
#   6 : Rossby-Haurvitz waves (shallow water)
#   8 : Unstable jet (shallow water)
case_number = 6

[Time_integration]

# Time step
dt = $dt  
#1800

# End time of the simulation in sec
t_end = $tf 
#1209600

# Time integration scheme
# Possible values  = tvdrk3 : 3th Order TVD-RK time integration
#                    epi2 : 2n order exponential propagation iterative
#                    epi3 : 3rd order exponential propagation iterative (Recommended)
#                    epirk4s3A : 4th order 3-stage EPIRK time integration
time_integrator = epi2
exponential_solver=exode
exode_method = $exode_method

# Solver tolerance
tolerance = $tol

gmres_restart = 20

starting_step = 0

[Preconditioning]
preconditioner = none
precond_tolerance = 1e-1
num_mg_levels = 3
num_pre_smoothe = 1
num_post_smoothe = 1
pseudo_cfl = 2.5e7
restrict_method = modal
kiops_dt_factor = 1.2
mg_solve_coarsest = 0
mg_smoother = erk1
precond_filter_apply = 0
verbose_precond = 1
exp_smoothe_spectral_radii = [3.0, 1.0, 0.5, 2.2, 1.0]

[Spatial_discretization]

# The grid will have (nbsolpts) x (nbsolpts) nodal points in each elements.
nbsolpts = $nbsolpts

# Number of element in x^1 and x^2 directions
# Each face of the cube have (nbElements x nbElements) elements, for a total of (6 x nbElements x nbElements) elements.
nb_elements_horizontal = $nb_elements

[Output_options]

# Print blockstats every \"stat_freq\" steps, 0 to disable.
stat_freq = 0

# Output solution every \"output_freq\" steps, 0 to disable.
output_freq = 0

# Save the state vector to a file at every \"save_state_freq\" steps. 0 to disable.
save_state_freq = 0

# Store statistics about the solver (iterations, residuals, etc.). 0 to disable.
store_solver_stats = 0

# Output directory
output_dir  = results


        " > $configDir/$configFile
	
        testoutputSubDir=${testoutputDir}/${nbsolpts}x${nb_elements}x${nb_elements}
        mkdir $testoutputSubDir
	testoutputFile=${exode_method}_case6_tol_${tol}_dt_${dt}_tf_${tf}_grid_${nbsolpts}x${nb_elements}x${nb_elements}_using_${core}_cores
        echo $testoutputFile
        begin=$(date +%s)	
        mpirun -n $core ./main_gef.py ./config/${configFile} > ${testoutputSubDir}/${testoutputFile}.txt
	end=$(date +%s)
	echo "Elapsed Time: $(($end-$begin)) seconds"

done
done
done

echo "Test completed!"
