ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_1k_1t.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                   \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_1k_as.bat      \
           -listing $(pwd)/listings



