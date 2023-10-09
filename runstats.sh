
ord_soumet -cpus 4704                                  \
           -w 180                                      \
           -jn gef_4704_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/wblockstats/call_p4704_wstats.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 10584                                 \
           -w 180                                      \
           -jn gef_10584_stats                         \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/wblockstats/call_p10k_wstats.bat     \
           -listing $(pwd)/listings



