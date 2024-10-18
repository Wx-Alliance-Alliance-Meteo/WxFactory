"""
	Setting up the parallel environment for solving stiff
	pdes using exponential integrators. 


	|-------|-------|-------|
	|	p0  |  p1   |   p2  |
	|-------|-------|-------|
	|	p3  |  p4   |   p5  | 
	|-------|-------|-------|
	|	p6  |  p7   |   p8  |
	|-------|-------|-------|
	|  p9   | p10   |  p11  |
	|-------|-------|-------|

	---> x increasing
	|
	v
	y increasing


	----processor coordinates----

    |-------|-------|-------|
	| (0,0) | (1,0) | (2,0) |
	|-------|-------|-------|
	| (0,1) | (1,1) | (2,1) |  
	|-------|-------|-------|
	| (0,2) | (1,2) | (2,2) |
	|-------|-------|-------|
	| (0,3) | (1,3) | (2,3) |
	|-------|-------|-------|


	the domain is split both along the x and y axis, each 
	processor will hold a chunk of the rows and columns.

	Unfortunately, when we stack the 2D domain into a 1D
	vector, this means the data will not be contiguous.


	---p0---
	|      | row 1
	|      |
	---p1---
	|      | row 1
	|      |
	---p2---
	|      | row 1
	|      |
	---p3---
	|      | row 1
	|      |
	---p0---
	|      | row 2
	|      |
	---p1---
	|      | row 2
	|      |
	---p2---
	|      | row 2
	|      |
	---p3---
	|      | row 2
	|      |
	--------

	..etc. 

	To run: mpirun -np p python3 laplacian_exp_perf.py m ortho_method

	where p is the number of processors and m is the dimension of the Krylov subspace
	and ortho_method is the orthogonalization technique

"""

import numpy as np 
import math 
import scipy.linalg
import sys 

from mpi4py	import MPI

class InitWorld:

	#constructor
	def __init__(yo, comm, BCType, domain, numPoints):

		yo.rank = comm.Get_rank()
		yo.size = comm.Get_size()

		yo.BCType = BCType
		
		if (comm.Get_rank() == 0):
			yo.IamRoot = True
		else:
			yo.IamRoot = False	


		yo.procs_per_xaxis = int(np.sqrt(yo.size))

		#if not square, then exit
		if yo.procs_per_xaxis**2 != yo.size:
			print("Wrong number of processors, must be square!")
			exit()	

		#else, should be divided evenly 
		else:
			#split up domain into squares 
			yo.procs_per_yaxis = yo.procs_per_xaxis
			yo.procEvenSplit   = True

		#rank to grid ID, for organizing domain
		yo.procYID = int (yo.rank / yo.procs_per_yaxis)
		yo.procXID = int (yo.rank % yo.procs_per_xaxis)

		#set up the arrays for local mat-vec communication

		#set id's of bottom neighboors
		#this is the global rank of the processors
		if yo.procYID == 0: #there are no bottom neighboors
			if yo.BCType == "periodic":
				yo.BottomNeighRank = yo.procs_per_xaxis*(yo.procs_per_yaxis-1) + yo.procXID
			else:
				yo.BottomNeighRank = None

		else: #get rank of processor below
			yo.BottomNeighRank = (yo.procYID-1)*yo.procs_per_xaxis + yo.procXID

		#set id's of top neighboors
		#this is the global rank of the processors
		if yo.procYID == yo.procs_per_yaxis -1: #last processor has no top neighboor
			if yo.BCType == "periodic":
				yo.TopNeighRank =  yo.rank - yo.procs_per_xaxis*(yo.procs_per_yaxis-1)
			else:
				yo.TopNeighRank = None

		else: #get rank of processor above 
			yo.TopNeighRank = (yo.procYID +1)*yo.procs_per_xaxis + yo.procXID

		#set rank of left neighboor
		#this saves the global rank of the processor for communication
		if yo.procXID == 0: #there is no left neighboor
			if yo.BCType == "periodic":
				yo.LeftNeighRank = yo.rank + (yo.procs_per_xaxis-1) #periodic bc's
			else:
				yo.LeftNeighRank = None 

		else: #get rank of left processor
			yo.LeftNeighRank = yo.rank - 1

		#set rank of right neighboor
		#note for the edge cases assume periodic bc
		#this saves global rank of proc. for communication
		if yo.procXID == yo.procs_per_xaxis -1:
			if yo.BCType == "periodic":
				yo.RightNeighRank = yo.rank - (yo.procs_per_xaxis-1) #periodic bc's
			else:
				yo.RightNeighRank = None

		else: #get rank of right processor
			yo.RightNeighRank = yo.rank + 1

		#Now set up the computational domain 
		#total number of points for axis
		yo.xL               = domain[0]
		yo.xR               = domain[1]
		yo.numPointswBound  = numPoints #total points including the boundary domain

		if BCType == "periodic":
			yo.numPointsPerAxis = yo.numPointswBound - 1 #remove only one end pt
		else:
			yo.numPointsPerAxis = yo.numPointswBound - 2 #remove both end points


		yo.totalPoints      = (yo.numPointsPerAxis)**2 #points in total domain, not per processor
		yo.numPointsX       = int(yo.numPointsPerAxis / yo.procs_per_xaxis) #points / processors

		#number of rows and columns per processor
		#if even, there will be an even chunk of the domain between each processor
		if yo.procEvenSplit :
			yo.numPointsY    = yo.numPointsX

		#this is for that special 8192 case 
		else: 
			yo.numPointsY = int(yo.numPointsPerAxis / yo.procs_per_yaxis)
			
		#total size when unrolled into a 1D array for each processor
		yo.oneDsize = yo.numPointsX*yo.numPointsY

		#dx
		yo.dx = (yo.xR - yo.xL) / (yo.numPointswBound -1)

		if BCType == "periodic":
			#start at the left boundary
			#here we remove the right point
			yo.startX = (yo.xL + (yo.dx * yo.procXID * yo.numPointsX))  
			yo.startY = (yo.xL + (yo.dx * yo.procYID * yo.numPointsY))
		else:
			#startX and startY grid point of each processor
			yo.startX = yo.xL + (yo.dx * yo.procXID * yo.numPointsX) + yo.dx
			yo.startY = yo.xL + (yo.dx * yo.procYID * yo.numPointsY) + yo.dx



