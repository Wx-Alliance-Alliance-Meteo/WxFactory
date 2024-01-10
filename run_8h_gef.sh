#----------------------srerk---------------------------------

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s6_1sv1.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s6_1sv2.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s6_1sv3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s3_1sv1.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s3_1sv2.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s3_1sv3.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_s3_1sv4.bat   \
           -listing $(pwd)/listings
