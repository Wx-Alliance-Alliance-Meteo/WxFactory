
#--epi---
ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_4k_1sv1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_4k_1sv2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_4k_nev1.bat      \
           -listing $(pwd)/listings



