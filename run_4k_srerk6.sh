
#----------------------4704---------------------------------

#for case 8
ord_soumet -cpus 4704                                   \
           -w 180                                       \
           -jn gef_4704_epi6                             \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/call_p4704_srerk6.bat      \
           -listing $(pwd)/listings


