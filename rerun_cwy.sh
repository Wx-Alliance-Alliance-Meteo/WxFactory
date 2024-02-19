
#--------------1k-----------------
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_rr_cwy_1k.bat      \
           -listing $(pwd)/listings

#-------------------2k----------------
ord_soumet -cpus 2646                                \
           -w 100                                    \
           -jn gef_2646                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_rr_cwy_2k.bat      \
           -listing $(pwd)/listings

#----------------4k-----------------------
ord_soumet -cpus 4704                                \
           -w 90                                    \
           -jn gef_4704                             \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_rr_cwy_4k.bat      \
           -listing $(pwd)/listings
