
#one time runs
ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_1t.bat      \
           -listing $(pwd)/listings

#alternative sizes
ord_soumet -cpus 10584                                \
           -w 180                                    \
           -jn gef_10584_altsizes                            \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_10k_as.bat      \
           -listing $(pwd)/listings
