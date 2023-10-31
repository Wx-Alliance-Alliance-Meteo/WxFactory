
#----------------------2646---------------------------------

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/kiopsstats/call_2646_kiopsstats_epi.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                              \
           -w 180                                  \
           -jn gef_2646                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/kiopsstats/call_2646_kiopsstats_srerk.bat  \
           -listing $(pwd)/listings

