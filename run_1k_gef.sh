ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_1176_nev2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 60                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_2k_1s.bat      \
           -listing $(pwd)/listings
