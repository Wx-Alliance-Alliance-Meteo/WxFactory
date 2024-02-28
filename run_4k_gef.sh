#----------------------4704---------------------------------
#---s6---
ord_soumet -cpus 4704                                \
           -w 90                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_nesr6.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_s6_1s.bat      \
           -listing $(pwd)/listings
#srerk3 
ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_s3_1s.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/call_4704_nesr3.bat      \
           -listing $(pwd)/listings

#--epi---
ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/final_run/call_epi_4k_1sv1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/final_run/call_epi_4k_1sv2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 4704                                \
           -w 180                                    \
           -jn gef_4704                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/final_run/call_epi_4k_nev1.bat      \
           -listing $(pwd)/listings



