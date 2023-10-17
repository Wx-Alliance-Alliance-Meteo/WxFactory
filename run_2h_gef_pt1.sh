
#----------------------294---------------------------------

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_294_epi.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                              \
           -w 180                                  \
           -jn gef_294                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_294_cwy1s.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s3c5.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s3c5nec.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 294                              \
           -w 180                                  \
           -jn gef_294                            \
           -mpi                                    \
           -share e                                \
           -jobfile bat_files/call_294_s3c5nei.bat  \
           -listing $(pwd)/listings

ord_soumet -cpus 294                               \
           -w 180                                   \
           -jn gef_294                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_294_s3c5ne.bat   \
           -listing $(pwd)/listings


