
#re-running some test because they didn't run the firs time.
#maybe they ran out of time?

#for case 5
ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p294_partA.bat \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p294_partB.bat \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p294_partA.bat \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p294_partB.bat \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p294_partA.bat \
           -listing $(pwd)/listings

ord_soumet -cpus 294                                    \
           -w 360                                       \
           -jn gef_294_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p294_partB.bat \
           -listing $(pwd)/listings

#-------------------------------------------------------

#for case 5
ord_soumet -cpus 864                                    \
           -w 360                                       \
           -jn gef_864_c5                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case5/call_p864.bat       \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 864                                    \
           -w 360                                       \
           -jn gef_864_c6                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case6/call_p864.bat       \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 864                                    \
           -w 360                                       \
           -jn gef_864_c8                               \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p864.bat       \
           -listing $(pwd)/listings

#---------------------------------------------------------

#for case 5
ord_soumet -cpus 1176                                  \
           -w 360                                      \
           -jn gef_1176_c5                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case5/call_p1176.bat     \
           -listing $(pwd)/listings

#for case 5
ord_soumet -cpus 2646                                  \
           -w 360                                      \
           -jn gef_2646_c5                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case5/call_p2646.bat     \
           -listing $(pwd)/listings


#for case 5
ord_soumet -cpus 4704                                  \
           -w 360                                      \
           -jn gef_4704_c5                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case5/call_p4704.bat     \
           -listing $(pwd)/listings

#---------------------------------------------------------

#for case 6
ord_soumet -cpus 1176                                  \
           -w 360                                      \
           -jn gef_1176_c6                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case6/call_p1176.bat     \
           -listing $(pwd)/listings

#for case 6
ord_soumet -cpus 2646                                  \
           -w 360                                      \
           -jn gef_2646_c6                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case6/call_p2646.bat     \
           -listing $(pwd)/listings


#for case 6
ord_soumet -cpus 4704                                  \
           -w 360                                      \
           -jn gef_4704_c6                             \
           -mpi                                        \
           -share e                                    \
           -jobfile bat_files/case6/call_p4704.bat     \
           -listing $(pwd)/listings

#----------------------------------------------------------

#for case 8
ord_soumet -cpus 1176                                   \
           -w 360                                       \
           -jn gef_1176_c8                              \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p1176.bat      \
           -listing $(pwd)/listings

#for case 8
ord_soumet -cpus 2646                                   \
           -w 360                                       \
           -jn gef_2646_c8                              \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p2646.bat      \
           -listing $(pwd)/listings


#for case 8
ord_soumet -cpus 4704                                   \
           -w 360                                       \
           -jn gef_4704_c8                              \
           -mpi                                         \
           -share e                                     \
           -jobfile bat_files/case8/call_p4704.bat      \
           -listing $(pwd)/listings









