#---epi6---
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 30                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_2k_1s.bat      \
           -listing $(pwd)/listings



