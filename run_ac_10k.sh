
ord_soumet -cpus 10000                                  \
           -w 180                                      \
           -jn ac_10000_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/ac/call_10k.bat     \
           -listing $(pwd)/listings



