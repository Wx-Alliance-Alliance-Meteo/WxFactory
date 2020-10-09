#!/usr/bin/env bash

case_file="test_case.ini"

RUN="/usr/bin/mpirun -H 127.0.0.1:6 -merge-stderr-to-stdout -tag-output -n 6 -cpu-list 0,1,1,2,2,3 python3 ./main_gef.py ${case_file}"

#sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=7/'
#${RUN}
sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=6/'
${RUN}
sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=5/'
${RUN}
sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=4/'
${RUN}
sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=3/'
${RUN}
#sed -i ${case_file} -e 's/nbsolpts=[0-9]/nbsolpts=8/'
#${RUN}


