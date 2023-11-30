#all the remaining epi calls: epi5 1s + epi 4,5,6 ne
#----------------------294---------------------------------

#-------------EPI5 1S-----------------
ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_1sv1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_1sv2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_1sv3.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_1sv4.bat      \
           -listing $(pwd)/listings

#---------------norm estimate vertions--------------------
#------------EPI 4 ne -----------------
ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_294_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_294_nev2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e4_294_nev3.bat      \
           -listing $(pwd)/listings

#-------------EPI5 ne-----------------
ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_nev2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e5_294_nev3.bat      \
           -listing $(pwd)/listings


#-------------EPI6 ne-----------------
ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_294_nev1.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_294_nev2.bat      \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                \
           -w 180                                    \
           -jn gef_294                         \
           -mpi                                      \
           -share e                                  \
           -jobfile bat_files/finalrun/call_e6_294_nev3.bat      \
           -listing $(pwd)/listings

