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
The python-cartopy can be installed for AUR.

# Executing the model

The allowed number of PEs is of the form $6*n^2$, for n> = 0.

Here is a sample command to run the model:
`mpirun -merge-stderr-to-stdout -tag-output -n 6 python3 ./main_gef.py config/case6.ini`
