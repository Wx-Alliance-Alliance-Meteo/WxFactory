
#----------------------2646---------------------------------

ord_soumet -cpus 2646                                \
           -w 180                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_2646_epi.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                              \
           -w 180                                  \
           -jn gef_2646                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_2646_nesr6.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_nesr3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 2646                               \
           -w 180                                   \
           -jn gef_2646                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_2646_srerk.bat   \
           -listing $(pwd)/listings




