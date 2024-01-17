
ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn burg_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_1600_epi4.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn burg_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_1600_epi5.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn burg_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_1600_epi6.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn burg_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_1600_s3.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn burg_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_1600_s6.bat     \
           -listing $(pwd)/listings

#---------------------------------------------------
ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn burg_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_2500_epi.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn burg_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/burg/call_2500_sne.bat     \
           -listing $(pwd)/listings

