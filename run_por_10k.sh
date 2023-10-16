
ord_soumet -cpus 10000                                  \
           -w 180                                      \
           -jn por_10000_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/por/call_10k.bat     \
           -listing $(pwd)/listings



