
ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn ac_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_2500_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn ac_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_2500_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn ac_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_2500_epi6.bat     \
           -listing $(pwd)/listings

