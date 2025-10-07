#! /bin/env bash

# Load code tools
echo "Loading tools"
# . r.load.dot rpn/code-tools/latest/env/rhel-8-amd64-64@gnu-9.3.0
. r.load.dot rpn/code-tools/latest/env/rhel-8-icelake-64@inteloneapi-2022.1.2

# Load environment (vma000)
echo "Loading environment"
# . ~/../vma000/python-environments/wx-gpsc/bin/activate
source /home/ngv000/sitestore5/python-environments/wx-gpsc/bin/activate

echo $PWD
# CONFIG_FILE=/home/ngv000/Documents/code/WxFactory/config/dcmip31.ini
# CONFIG_FILE=/home/ngv000/repos/WxFactory/tests/data/integration/dcmip31/config.ini
CONFIG_FILE=/home/ngv000/repos/WxFactory/config/dcmip31.ini
WX_DIR=/home/ngv000/repos/WxFactory


echo "Config file: ${CONFIG_FILE}"
echo "Wx dir: ${WX_DIR}"

echo "Starting run"
echo "${WX_DIR}/WxFactory"
mpirun -n 24 python3 ${WX_DIR}/WxFactory ${CONFIG_FILE}
# python3 ${WX_DIR}/WxFactory ${CONFIG_FILE}