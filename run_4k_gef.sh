#------------run epi 4 5 6 again--------------

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
           -jobfile bat_files/finalrun/call_epi_4k_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_4k_nev2.bat      \
           -listing $(pwd)/listings



#----------------------4704---------------------------------

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_s6_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_s3_1s.bat      \
           -listing $(pwd)/listings


ord_soumet -cpus 4704                              \
           -w 180                                  \
           -jn gef_4704                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_4704_nesr6.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                               \
           -w 180                                   \
           -jn gef_4704                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_4704_nesr3.bat   \
           -listing $(pwd)/listings
