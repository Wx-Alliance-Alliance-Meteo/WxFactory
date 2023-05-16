

#----------------------2646---------------------------------
#for case 5 and case 6
ord_soumet -cpus 2646                                  \
           -w 180                                      \
           -jn gef_2646_epi2                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/call_p2646_epi2.bat     \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 2646                                  \
           -w 180                                      \
           -jn gef_2646_c8                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case8/epi2/call_p2646.bat     \
           -listing $(pwd)/listings


