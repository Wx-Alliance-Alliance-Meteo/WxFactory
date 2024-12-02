
#--epi---
ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_4k_1t.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_4k_as.bat      \
           -listing $(pwd)/listings
