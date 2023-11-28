#----------------epi 4 5 6 again-------------------------

#-----------epi4 --------------------
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_1176_1s.bat      \
           -listing $(pwd)/listings

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

#----------epi 5---------------------
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_nev2.bat      \
           -listing $(pwd)/listings



#----------epi 6---------------------
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_nev2.bat      \
           -listing $(pwd)/listings

#----------------------srerk---------------------------------

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_1sv1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_1sv2.bat  \
           -listing $(pwd)/listings


ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_1sv3.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_nev1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_nev2.bat  \
           -listing $(pwd)/listings


ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s3_nev3.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_1sv1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_1sv2.bat  \
           -listing $(pwd)/listings


ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_1sv3.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_nev1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_nev2.bat  \
           -listing $(pwd)/listings


ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_s6_nev3.bat  \
           -listing $(pwd)/listings
