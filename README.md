The name of the project, GEF, is a French acronym which stands for ***G**EM en **É**léments **F**inis*.  

# Requirements

GEF was built for Python3.  It also requires an MPI implementation.

## Python packages
* `numpy` Scientific tools for Python
* `scipy` Python-based ecosystem of open-source software for mathematics, science, and engineering
* `mpi4py` Python interface for MPI
* `netcdf4-openmpi` Python/NumPy interface to the netCDF C library (openmpi version)

## Optional
* `mayavi` Visualization toolkit
* `matplotlib` A python plotting library, making publication quality plots
* `cartopy` A cartographic python library with matplotlib support for visualisation

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
conda install numpy scipy mpi4py netcdf4=*=mpi*
```

If you want the OpenMPI implementation of MPI:
```
conda install mpi=*=openmpi
```

If you want the visualization capabilities of GEF (and do not mind a bigger
conda environment):
```
conda install mayavi matplotlib cartopy
```

To visualize profiles from python applications, install `snakeviz`:
```
conda install snakeviz
```

## XC50 clusters
Most necessary python modules are already available in an environment module. 
`netCDF4` is missing, so don't try to write the output without installing it.
`mpi4py` is not working in that environment, so you need to install it manually (and locally):

1. Load the python module
```
module load cray-python/#.#.#
```
2. Create a virtual environment, and activate it
```
python -m venv [path to env directory] --system-site-packages
source [path to env directory]/bin/activate
```
3. Download the mpi4py package from https://mpi4py.readthedocs.io/en/stable/install.html
2. Unpack the tarball, go to into the directory
3. Compile the module, specifying the mpi compiler
```
python setup.py build --mpicc=cc
```
4. Install it (will be installed in the virtual environment)
```
python setup.py install
```

### Running on XC50
Before launching the program, don't forget to load the python module and activate the
virtual environment where `mpi4py` has been installed. Additionally, you need to set
the locale to UTF-8.
```
module load cray-python/#.#.#
source [virtual environment directory]/bin/activate
export LANG=en_CA.UTF-8
```

# Executing the model

The allowed number of PEs is of the form $6 n^2$, for $n \ge 0$.

Here is a sample command to run the model:
`mpirun -merge-stderr-to-stdout -tag-output -n 6 python3 ./main_gef.py config/case6.ini`


## Profiling GEF

You can generate an execution profile when running GEF by adding the `--profile` flag to the main command. For example:
```
mpirun -n 6 python3 ./main_gef.py --profile config/case6.ini
```

This will generate a set of `profile_####.out` files, one for each launched process, that can be viewed with `snakeviz`. _You need to be able to open a browser window from the terminal to use this command_:
```
snakeviz ./profile_0000.out
```