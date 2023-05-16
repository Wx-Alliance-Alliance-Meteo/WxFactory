
#----------------------4704---------------------------------

#for case 8
ord_soumet -cpus 4704                                   \
           -w 180                                       \
           -jn gef_4704_epi6                             \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/call_p4704_srerk3.bat      \
           -listing $(pwd)/listings


#----------------------10584---------------------------------
#for case 5
ord_soumet -cpus 10584                                  \
           -w 180                                      \
           -jn gef_10584_epi6                            \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p10584_srerk3.bat     \
           -listing $(pwd)/listings



