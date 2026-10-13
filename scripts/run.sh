#!/usr/bin/env bash

mpi_exports=()
while IFS='=' read -r variable value; do
    case "$variable" in
        BASH_*|CUDA_*) continue ;;
    esac
    case "$variable" in
        [a-zA-Z0-9]*) ;;
        *) continue ;;
    esac
    case "$value" in
        *\'*|*\"*) continue ;;
    esac
    mpi_exports+=(-x "$variable")
done < <(env)

mpi_exports+=(-x LD_PRELOAD=libmpi.so)
exec mpirun "${mpi_exports[@]}" "$@"
