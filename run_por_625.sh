
ord_soumet -cpus 625                                  \
           -w 180                                      \
           -jn por_625_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_625_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 625                                  \
           -w 180                                      \
           -jn por_625_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_625_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 625                                  \
           -w 180                                      \
           -jn por_625_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_625_epi6.bat     \
           -listing $(pwd)/listings

