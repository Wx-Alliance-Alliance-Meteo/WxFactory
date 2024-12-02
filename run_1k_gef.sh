ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_1t.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                   \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_as.bat      \
           -listing $(pwd)/listings



