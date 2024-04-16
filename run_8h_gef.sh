#---epi 4---
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e4_864_nev2.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e4_864_nev1.bat   \
           -listing $(pwd)/listings

#---epi 5---
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e5_864_nev1.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e5_864_nev2.bat   \
           -listing $(pwd)/listings

#---epi 6---
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e6_864_nev1.bat   \
           -listing $(pwd)/listings

ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/finalrun/call_e6_864_nev2.bat   \
           -listing $(pwd)/listings

#----------------------srerk---------------------------------

#s6 c5
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt1.bat   \
           -listing $(pwd)/listings

#s6 c6
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt2.bat   \
           -listing $(pwd)/listings

#s3 c5
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt3.bat   \
           -listing $(pwd)/listings

#s3 c6
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt4.bat   \
           -listing $(pwd)/listings

#s6 c8
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt5.bat   \
           -listing $(pwd)/listings

#s3 c8
ord_soumet -cpus 864                               \
           -w 180                                   \
           -jn gef_864                             \
           -mpi                                     \
           -share e                                 \
           -jobfile bat_files/call_864_srpt6.bat   \
           -listing $(pwd)/listings

