

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_2k_1t.bat      \
           -listing $(pwd)/listings


ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_2k_as.bat      \
           -listing $(pwd)/listings

