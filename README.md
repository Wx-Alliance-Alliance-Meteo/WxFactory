The name of the project, GEF, is a French acronym which stands for ***G**EM en **É**léments **F**inis*.  

# Requirements

GEF was built for Python3.  It also requires an MPI implementation.

## Python packages
* `numpy` Scientific tools for Python
* `scipy` Python-based ecosystem of open-source software for mathematics, science, and engineering
* `sympy` Python library for symbolic mathematics
* `mpi4py` Python interface for MPI
* `netcdf4` Python/NumPy interface to the netCDF C library (MPI version)
* `matplotlib` A python plotting library, making publication quality plots

## Other libraries
* `netcdf4` Library to handle netCDF files. **Must be an MPI version of it**
* `sqlite` To be able to store solver stats.

## Optional
* `mayavi` Visualization toolkit
* `cartopy` A cartographic python library with matplotlib support for visualisation
* `tqdm`   Progress bar when generating matrices
* `more-itertools` For using Bamphi solver

Python packages can be installed with the package management system of your
Linux distribution or with `pip`.  A few distribution specific instructions
are given below.

## Ubuntu 18.04
Here are the commands that need to be executed in order to install the
dependencies on Ubuntu 18.04:
```
pip3 install --user mpi4py
pip3 install --user PyQt5==5.14.0
pip3 install --user mayavi
```

## ArchLinux
```
sudo pacman -S python-numpy python-scipy python-mpi4py python-netcdf4-openmpi mayavi python-matplotlib 
```
The python-cartopy package can be installed from the AUR.

## Conda
The necessary packages are available from the conda-forge channel, so it should
be added to the list of default channels for easier use of the various commands
```
conda config --add channels conda-forge
conda create -n gef "python>=3.8"
conda activate gef
conda install numpy scipy sympy mpi4py matplotlib
# NetCDF stuff (in general):
conda install netcdf4=*=mpi*
```

To be able to use the system MPI library on Robert/Underhill:
```
conda install netcdf4=*=mpi_mpich_* mpich=3.3.*=external_*
```

If you want the OpenMPI implementation of MPI (on other systems):
```
conda install mpi=*=openmpi
```

If you want the visualization capabilities of GEF (and do not mind a bigger
conda environment):
```
conda install mayavi cartopy
```

To visualize profiles from python applications, install `snakeviz`:
```
conda install snakeviz
```

## Running GEF

```
# With the cubed sphere as a grid:
mpirun -n 6 ./main_gef.py config/case6.ini

# With the 2D cartesian grid:
./main_gef.py config/gaussian_bubble.ini
```

## Profiling GEF

You can generate an execution profile when running GEF by adding the `--profile` flag to the main command. For example:
```
mpirun -n 6 python3 ./main_gef.py --profile config/case6.ini
```

This will generate a set of `profile_####.out` files, one for each launched process, that can be viewed with `snakeviz`. _You need to be able to open a browser window from the terminal to use this command_:
```
snakeviz ./profile_0000.out
```

## 2D test cases
Here is an example of a command to run the model for the André Robert bubble test case:
```
python main_bubble.py config/gaussian_bubble.ini
```