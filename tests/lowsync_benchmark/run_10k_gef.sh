
#one time runs
ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_10k_dob.bat      \
           -listing $(pwd)/listings

#alternative sizes
ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584_altsizes                            \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_10k_n14.bat      \
           -listing $(pwd)/listings


#alternative sizes
ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584_altsizes                            \
           -mpi                                      \
           -share e                                  \
           -jobfile tests/lowsync_benchmark/bat_files/2024_12/call_10k_n28.bat      \
           -listing $(pwd)/listings

