#!/usr/bin/env bash

vars=$(env                      \
        | grep -v "[']"         \
        | grep -v '"'           \
        | grep -v "^BASH_"      \
        | grep -v "^CUDA_"      \
        | grep "^[a-zA-Z0-9]"   \
        | sed -e 's/=.*$//')

exports=""
for v in $vars; do
    exports="${exports} -x $v"
done

exports="${exports} -x LD_PRELOAD=libmpi.so"

mpirun ${exports} $@
