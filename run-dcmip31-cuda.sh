#!/bin/sh

OMPI_MCA_opal_cuda_support=true \
	CUDA_VISIBLE_DEVICES=1,2,3 \
	mpiexec -n 6 \
	python main_gef.py config/dcmip31_cuda.ini

