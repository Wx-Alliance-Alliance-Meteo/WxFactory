#----------------run epi 4 5 6 again------------------


#----------epi4 ------------
ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_2k_1s.bat      \
           -listing $(pwd)/listings


#----------epi5 ------------
ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_2k_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_2k_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_2k_nev2.bat      \
           -listing $(pwd)/listings

#----------epi6 ------------
ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_2k_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_2k_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_2k_nev2.bat      \
           -listing $(pwd)/listings

#----------------------srerk---------------------------------

ord_soumet -cpus 2646                              \
           -w 180                                  \
           -jn gef_2646                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_2646_nesr6v1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                              \
           -w 180                                  \
           -jn gef_2646                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_2646_nesr6v2.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_nesr3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_s3_1s.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_s6_1sv1.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_s6_1sv2.bat   \
           -listing $(pwd)/listings
