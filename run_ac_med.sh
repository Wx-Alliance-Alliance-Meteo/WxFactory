
ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn ac_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_1600.bat     \
           -listing $(pwd)/listings

#---------------------------------------------------
ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn ac_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_2500.bat     \
           -listing $(pwd)/listings


