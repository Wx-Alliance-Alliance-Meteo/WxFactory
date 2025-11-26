#! /bin/env bash

# Fill
CONFIG_FILE=.../WxFactory/config/dcmip31.ini
WX_DIR=.../WxFactory


echo "Config file: ${CONFIG_FILE}"
echo "Weather factory dir: ${WX_DIR}"

echo "Starting run"
mpirun -n 6 -bind-to none python3 ${WX_DIR}/WxFactory ${CONFIG_FILE}
srun -n 6 python3 ${WX_DIR}/WxFactory ${CONFIG_FILE}