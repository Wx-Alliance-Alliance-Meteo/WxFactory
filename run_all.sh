

#for case 5
ord_soumet -cpus 2646                                    \
           -w 180                                       \
           -jn gef_2646_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p2646.bat       \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 2646                                    \
           -w 180                                       \
           -jn gef_2646_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p2646.bat       \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 2646                                    \
           -w 180                                       \
           -jn gef_2646_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p2646.bat       \
           -listing $(pwd)/listings


