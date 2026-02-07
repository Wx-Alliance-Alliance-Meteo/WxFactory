#############################################################
# Container build command:
# docker build --platform=linux/amd64 -t wxfactory .
#
# Container start and run the interactive bash session inside:
# docker run -it --platform=linux/amd64 --mount type=bind,src=/root/kate/work,dst=/work --rm wxfactory /bin/bash
# 
# Note: above command mounts host directory /root/kate/work inside the container as /work
#       All the changes done to the file in a host env will be visible in guest, and in reverse
#
#############################################################
# In the container shell, run commands as:
#
# mpirun --allow-run-as-root -n 6 ./WxFactory config/case6.ini 
#
# ./WxFactory config/gaussian_bubble.ini
#
FROM python:3.12

# Set environment variables 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt-get update
RUN apt-get -y install zip git
RUN apt-get -y install openmpi-bin openmpi-common libopenmpi-dev

# Clone from Git
# RUN git clone https://github.com/Wx-Alliance-Alliance-Meteo/WxFactory.git /app

# Install pyhton requirements
RUN python -m pip install --upgrade setuptools pip
RUN python -m pip install --upgrade numpy scipy sympy mpi4py pybind11 netcdf4 matplotlib
RUN python -m pip install --upgrade cartopy tqdm snakeviz

# Set workdir where WxFactory cloned
WORKDIR /