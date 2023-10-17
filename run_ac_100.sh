
ord_soumet -cpus 100                                  \
           -w 180                                      \
           -jn ac_100_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_100_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 100                                  \
           -w 180                                      \
           -jn ac_100_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_100_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 100                                  \
           -w 180                                      \
           -jn ac_100_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_100_epi6.bat     \
           -listing $(pwd)/listings

