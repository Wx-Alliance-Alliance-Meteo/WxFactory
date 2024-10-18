
# Example test launcher

SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LISTING_DIR=${SELF_DIR}/../../listings

# script_files="bat_files/finalrun/call_e4_294_k.bat bat_files/call_294_s6c8ne.bat bat_files/call_294_s6c8nei.bat bat_files/call_294_s6c8nec.bat"
script_files="bat_files/finalrun/call_e4_294_k.bat"

mkdir -pv ${LISTING_DIR}
for s in ${script_files}; do
    # Indicate correct WxFactory directory in script
    s_tmp="${s}.tmp"
    sed -e 's|${WX_FACTORY_DIR}|'${SELF_DIR}/../..'|' < ${s} > ${s_tmp}

    # Submit job
    ord_soumet -cpus 294                               \
            -w 180                                   \
            -jn gef_294                             \
            -mpi                                     \
            -share e                                 \
            -jobfile ${s_tmp}                         \
            -listing ${LISTING_DIR}

    # Remove tmp script file right after submission
    rm -f ${s_tmp}
done
