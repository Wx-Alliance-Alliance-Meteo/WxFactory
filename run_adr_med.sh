
ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_epi.bat     \
           -listing $(pwd)/listings


ord_soumet -cpus 1600                                  \
           -w 180                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_sr.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 1600                                  \
           -w 120                                      \
           -jn adr_1600_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_1600_sne.bat     \
           -listing $(pwd)/listings

#---------------------------------------------------
ord_soumet -cpus 2500                                  \
           -w 120                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500_s3ne.bat     \
           -listing $(pwd)/listings

ord_soumet -cpus 2500                                  \
           -w 180                                      \
           -jn adr_2500_stats                          \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/adr/call_2500_s6ne.bat     \
           -listing $(pwd)/listings
