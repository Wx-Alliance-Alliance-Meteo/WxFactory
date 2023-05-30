
#----------------------4704---------------------------------
#for all cases with different integrators 
ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_epi4                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_epi4.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_epi6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_epi6.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_epi2                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_epi2.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_srerk3                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_srerk3.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_srerk6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_srerk6.bat     \
           -listing $(pwd)/listings



ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_epi2                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_epi2_partB.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_srerk3                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_srerk3_partB.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_srerk6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p4704_srerk6_partB.bat     \
           -listing $(pwd)/listings
