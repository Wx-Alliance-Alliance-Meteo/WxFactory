
ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_epi6.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500_epi2.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500_epi4.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500_epi6.bat     \
           -listing $(pwd)/listings



