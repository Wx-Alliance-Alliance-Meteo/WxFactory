The name of the project, GEF, is a French acronym which stands for ***G**EM en **É**léments **F**inis*.  

# Requirements

GEF was built for Python 3.11 (at least).  It also requires an MPI implementation.

## Python packages
* Python version at least 3.11
* `numpy` Scientific tools for Python
* `scipy` Python-based ecosystem of open-source software for mathematics, science, and engineering
* `sympy` Python library for symbolic mathematics
* `mpi4py` Python interface for MPI
* `netcdf4` Python/NumPy interface to the netCDF C library (MPI version)
* `matplotlib` A python plotting library, making publication quality plots

## Other libraries
* `netcdf4` Library to handle netCDF files. There is an MPI version of it, if you want parallel output
* `sqlite` To be able to store solver stats.

## Optional
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
```

## ArchLinux
```
sudo pacman -S python-numpy python-scipy python-mpi4py python-netcdf4-openmpi python-matplotlib 
```
The python-cartopy package can be installed from the AUR.

## Conda
The necessary packages are available from the conda-forge channel, so it should
be added to the list of default channels for easier use of the various commands
```
conda config --add channels conda-forge
conda create -n gef "python>=3.11"
conda activate gef
conda install numpy scipy sympy mpi4py matplotlib netcdf4
# On the Science network, we need to use the already-installed MPI library:
conda install numpy scipy sympy mpi4py matplotlib netcdf4 mpi=*=mpich
```

To be able to use NetCDF in parallel (for faster writing to disk) [Optional]
```
conda install netcdf4=*=mpi*
```

If you want the visualization capabilities of GEF:
```
conda install cartopy
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
python3 main_gef.py config/gaussian_bubble.ini
```