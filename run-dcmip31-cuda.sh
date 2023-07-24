#!/bin/sh

OMPI_MCA_opal_cuda_support=true \
	CUDA_VISIBLE_DEVICES=1,2,3 \
	mpiexec -n 6 \
	python -m kernprof -l main_gef.py config/dcmip31_cuda.ini

python -m line_profiler main_gef.py.lprof > main_gef.py.lprof.txt
