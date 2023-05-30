
#----------------------10584---------------------------------
#for all cases with different integrators 
ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_epi4                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_epi4.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_epi6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_epi6.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_epi2                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_epi2.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_srerk3                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_srerk3.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_srerk6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_srerk6.bat     \
           -listing $(pwd)/listings



ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_epi2                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_epi2_partB.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_srerk3                           \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_srerk3_partB.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_srerk6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_srerk6_partB.bat     \
           -listing $(pwd)/listings
