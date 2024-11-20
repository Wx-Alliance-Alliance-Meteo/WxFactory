"""
	function to print the final solution and stats to file

"""

import mpi4py.MPI
import numpy as np
import sys


# print finalSol, here, we expect sol to be the gathered
# array, so only one processor will call this function
# need world to acces variables like number of x-poins
# per processor, etc
def print_sol(finalSol, filename, world):

    finalSolOrder = np.zeros(world.totalPoints)

    for j in range(0, world.procs_per_yaxis):
        for k in range(0, world.numPointsY):
            for el in range(0, world.procs_per_xaxis):
                for m in range(0, world.numPointsX):

                    # a. what is the index of the large finalSol array
                    indxSol = (
                        el * world.numPointsX
                        + world.numPointsPerAxis * k
                        + j * world.numPointsX * world.numPointsY * world.procs_per_xaxis
                        + m
                    )

                    # b. what is the index of the processor data in finalSol
                    indxProc = j * world.procs_per_xaxis + el

                    # c. what is the index of the array elements in the finalSol list
                    indxArr = k * world.numPointsX + m

                    # print("j = {}, k = {}, el = {}, m = {}".format(j,k,el,m))
                    # print("indxSol = {}, indxProc = {}, indxArr = {}".format(indxSol, indxProc, indxArr))
                    # print("-------------------------------------------")

                    finalSolOrder[indxSol] = finalSol[indxProc][indxArr]

    with open(filename, "a") as gg:
        for j in range(0, len(finalSolOrder)):

            gg.write("{} \n".format(finalSolOrder[j]))

    return finalSolOrder


def print_stats(stats, filename, m):

    # create a file per method to save stats
    # in order: runtime , krylov error, loss of ortho

    with open(filename, "a") as gg:
        gg.write("{} {} {} {}\n".format(m, stats[2], stats[0], stats[1]))

    with open(filename, "a") as gg:
        for j in range(0, len(finalSolOrder)):

            gg.write("{} \n".format(finalSolOrder[j]))
