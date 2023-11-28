

ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_10k_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_epi_10k_ne.bat      \
           -listing $(pwd)/listings


ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_s3_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_s3_ne.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_s6_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_s6_ne.bat      \
           -listing $(pwd)/listings



