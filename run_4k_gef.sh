
#--epi---
ord_soumet -cpus 4704                                \
           -w 40                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_4k_1sv1.bat      \
           -listing $(pwd)/listings

