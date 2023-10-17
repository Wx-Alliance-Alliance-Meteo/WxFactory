
ord_soumet -cpus 400                                  \
           -w 180                                      \
           -jn ac_400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_400_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 400                                  \
           -w 180                                      \
           -jn ac_400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_400_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 400                                  \
           -w 180                                      \
           -jn ac_400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_400_epi6.bat     \
           -listing $(pwd)/listings

