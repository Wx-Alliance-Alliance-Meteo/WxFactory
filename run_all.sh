

#for case 5
ord_soumet -cpus 4704                                    \
           -w 180                                       \
           -jn gef_4704_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p4704.bat       \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 4704                                    \
           -w 180                                       \
           -jn gef_4704_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p4704.bat       \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 4704                                    \
           -w 180                                       \
           -jn gef_4704_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p4704.bat       \
           -listing $(pwd)/listings


