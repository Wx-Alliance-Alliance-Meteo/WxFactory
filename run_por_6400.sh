
ord_soumet -cpus 6400                                  \
           -w 180                                      \
           -jn por_6400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_6400_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 6400                                  \
           -w 180                                      \
           -jn por_6400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_6400_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 6400                                  \
           -w 180                                      \
           -jn por_6400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_6400_epi6.bat     \
           -listing $(pwd)/listings

