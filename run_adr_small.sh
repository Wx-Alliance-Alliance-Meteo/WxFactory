

ord_soumet -cpus 100                                  \
           -w 180                                      \
           -jn adr_100_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_100_s6.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 100                                  \
           -w 180                                      \
           -jn adr_100_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_100_s6_partB.bat     \
           -listing $(pwd)/listings

#-----------------------------------------------------------
ord_soumet -cpus 400                                  \
           -w 180                                      \
           -jn adr_400_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_400_s3.bat     \
           -listing $(pwd)/listings




