

#for case 5
ord_soumet -cpus 864                                    \
           -w 180                                       \
           -jn gef_864_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p864.bat       \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 864                                    \
           -w 180                                       \
           -jn gef_864_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p864.bat       \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 864                                    \
           -w 180                                       \
           -jn gef_864_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p864.bat       \
           -listing $(pwd)/listings


