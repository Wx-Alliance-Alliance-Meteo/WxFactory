

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_2k_1t.bat      \
           -listing $(pwd)/listings


ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_2k_as.bat      \
           -listing $(pwd)/listings

