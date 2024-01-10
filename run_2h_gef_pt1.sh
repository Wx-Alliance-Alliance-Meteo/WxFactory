
#remaining epi calls: these are just calls that didnt finish the full 
#7 runs, so missing a like 1 or 2 runs

ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e4_294_k.bat   \
           -listing $(pwd)/listings


#srerk 6 galewsky pmex ne
ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c8ne.bat   \
           -listing $(pwd)/listings

#srerk 6 galewsky icwyne 
ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c8nei.bat   \
           -listing $(pwd)/listings

#srerk 6 galewsky cwy ne
ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c8nec.bat   \
           -listing $(pwd)/listings


#srerk 3 cwyne case 5 (need 2 more)
ord_soumet -cpus 294                               \
           -w 90                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s3c5ne.bat   \
           -listing $(pwd)/listings









