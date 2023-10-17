
#----------------------4704---------------------------------

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_all.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                              \
           -w 180                                  \
           -jn gef_4704                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_4704_nesr6.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                               \
           -w 180                                   \
           -jn gef_4704                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_4704_nesr3.bat   \
           -listing $(pwd)/listings
