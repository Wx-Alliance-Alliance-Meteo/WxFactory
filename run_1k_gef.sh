
#----------------------1176---------------------------------

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_epi.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_srpt1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                               \
           -w 180                                   \
           -jn gef_1176                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_1176_srpt2.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                               \
           -w 180                                   \
           -jn gef_1176                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_1176_srpt3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                              \
           -w 180                                  \
           -jn gef_1176                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_1176_srpt4.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                               \
           -w 180                                   \
           -jn gef_1176                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_1176_srpt5.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                               \
           -w 180                                   \
           -jn gef_1176                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_1176_srpt6.bat   \
           -listing $(pwd)/listings



