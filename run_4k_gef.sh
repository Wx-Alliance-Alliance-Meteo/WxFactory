
ord_soumet -cpus 1176                                \
           -w 60                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_nev2.bat      \
           -listing $(pwd)/listings
#---srerk---
ord_soumet -cpus 4704                                \
           -w 90                                     \
           -jn gef_4704                              \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_nesr3.bat    \
           -listing $(pwd)/listings
