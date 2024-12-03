

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_2k_dob.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_2k_n14.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_2k_n28.bat      \
           -listing $(pwd)/listings

