
#remaining epi calls: these are just calls that didnt finish the full 
#7 runs, so missing a like 1 or 2 runs

ord_soumet -cpus 294                                \
           -w 120                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 120                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_nev2.bat      \
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

#srerk 6 galwsky cwy 1s (need 3 more runs)
ord_soumet -cpus 294                               \
           -w 120                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c8k.bat   \
           -listing $(pwd)/listings

#srerk 6 case 6 cwyne (need 1 more)
ord_soumet -cpus 294                               \
           -w 120                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c6nec.bat   \
           -listing $(pwd)/listings

#srerk 6 case 5 cwyne (need 1 more)
ord_soumet -cpus 294                               \
           -w 120                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s6c5ne.bat   \
           -listing $(pwd)/listings

#srerk 3 cwyne case 5 (need 2 more)
ord_soumet -cpus 294                               \
           -w 90                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s3c5ne.bat   \
           -listing $(pwd)/listings









