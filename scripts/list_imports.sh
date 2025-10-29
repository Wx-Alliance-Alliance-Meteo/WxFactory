#!/usr/bin/env bash

DIR=${1}

if [ $# -lt 1 ] || [ ! -d ${DIR} ]; then
    echo "Need to pass a directory"
    exit 1
fi

IGNORE="
abc\|\
ast\|\
collections\|\
configparser\|\
copy\|\
cupy\|\
cupyx\|\
cupy_backends\|\
functools\|\
georef\|\
glob\|\
hashlib\|\
importlib\|\
itertools\|\
json\|\
math\|\
matplotlib\|\
mpi4py\|\
netCDF4\|\
numpy\|\
operator\|\
os\|\
pandas\|\
pickle\|\
pybind11\|\
re\|\
rmn\|\
scipy\|\
setuptools\|\
shutil\|\
sqlite3\|\
sympy\|\
sys\|\
time\|\
tqdm\|\
types\|\
typing"

#grep -rw import ${DIR} | grep -vw ${IGNORE}

# for file in $(find ${DIR} -type f); do
#     grep -w import ${file} | grep -vw ${IGNORE}
# done

for dir in $(find ${DIR}/* -maxdepth 1 -type d); do
    echo -e "\n${dir}"
    # for file in $(find ${dir} -type f); do
        grep -rwh import ${dir} | grep -vw ${IGNORE} | grep -v '^ *#' | grep -v ' \.' | \
            sed -e 's/^\s*/  /' -e 's/from\s*\([^ ]*\)\s*import.*$/\1/' -e 's/\..*$//' | \
            sort -u
    # done
done
