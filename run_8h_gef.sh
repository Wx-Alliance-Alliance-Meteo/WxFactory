
#----------------------864---------------------------------

ord_soumet -cpus 864                                \
           -w 180                                    \
           -jn gef_864                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_864_epi.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 864                              \
           -w 180                                  \
           -jn gef_864                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_864_srpt1.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt2.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                              \
           -w 180                                  \
           -jn gef_864                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_864_srpt4.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt5.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt6.bat   \
           -listing $(pwd)/listings



