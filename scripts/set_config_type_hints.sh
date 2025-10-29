#!/usr/bin/env bash

SCRIPT_NAME=$(basename ${BASH_SOURCE[0]})
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=$(dirname ${SCRIPT_DIR})

config_code=${WX_DIR}/wx_factory/common/configuration.py
tmp_code=${config_code}.tmp

if [ ! -f ${config_code} ]; then
    echo "Could not locate configuration code at ${config_code}"
    exit -1
fi

hint_string=$(              \
    ${SCRIPT_DIR}/wx_config.py --list-hints | \
    grep -v 'running build_ext' | \
    sort | \
    sed -e 's/cs-str/str/' \
        -e 's/^/    /' \
)

sed -e '/--- START type hints ---/q' ${config_code} > ${tmp_code}
echo -e "${hint_string}" >> ${tmp_code}
sed -ne '/--- END type hints ---/,$ p' ${config_code} >> ${tmp_code}

mv ${tmp_code} ${config_code}
