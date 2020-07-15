The name of the project, GEF, is a French acronym which stands for ***G**EM en **É**léments **F**inis*.  

# Requirements

GEF was built for Python3.  It also requires an MPI implementation.

## Python packages
* `mayavi` Visualization toolkit
* `mpi4py` Python interface for MPI
* `scipy` Python-based ecosystem of open-source software for mathematics, science, and engineering

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
sudo pacman -S python-mpi4py python-scipy mayavi
```

# Executing the model

Since there is one PE allocated per face, GEF needs 6 PEs.

Here is a sample command to run the model:
`mpirun -merge-stderr-to-stdout -tag-output -n 6 python3 ./main_gef.py config/case6.ini`
