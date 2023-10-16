
ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn por_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_1600_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn por_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_1600_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn por_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_1600_epi6.bat     \
           -listing $(pwd)/listings

