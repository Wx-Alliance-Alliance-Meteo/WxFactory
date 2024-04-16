#---epi4---
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_1176_nev2.bat      \
           -listing $(pwd)/listings


#---epi5---
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_1176_nev2.bat      \
           -listing $(pwd)/listings

#---epi6---
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_1176_nev2.bat      \
           -listing $(pwd)/listings

#----srerk3------
#pmex + c5k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s3_nev1.bat      \
           -listing $(pwd)/listings

#icwy + c6k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s3_nev2.bat      \
           -listing $(pwd)/listings

#cwy + c8k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s3_nev3.bat      \
           -listing $(pwd)/listings


#----srerk6------
#pmex + c5k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s6_nev1.bat      \
           -listing $(pwd)/listings

#icwy + c6k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s6_nev2.bat      \
           -listing $(pwd)/listings

#cwy + c8k
ord_soumet -cpus 1176                                \
           -w 180                                    \
           -jn gef_1176                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_1176_s6_nev3.bat      \
           -listing $(pwd)/listings
