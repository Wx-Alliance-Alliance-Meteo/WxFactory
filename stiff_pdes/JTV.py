
"""
 Here I set up the discretization operators that will be used for the 
 RHS and Jtv functions for the pdes.

 There are 5 laplacian and 5 advection operators, each with the following BCs:
 1. homogeneous Dirichlet
 2. homogeneous Neumann
 3. Periodic

 Another Laplacian and advection operators are used for the 
 non-constant Jacobian. For example, the nonLinLapPeriodic is for the 
 Porous Medium equation, derivative of (u^2)_xx.

 And finally a pair for the rhs of the nonlinear advection or Laplacian.
 For example 1/2(u^2)_x rhs term in Burger's equation.

 There are also local communication helper functions to communicate local
 data for the stencils that are located on different processors. 

"""

import math
import numpy as np 
import mpi4py.MPI

#functions for communication
def sendData(vec, neigh, world):

	if neigh == "top":

		#1. send top data, j+1 index
		#if the rank is >0, send top data to previous rank
		#this will be the u_{i,j+1} data 
		#its the first vector the processor has access to

		if world.procYID > 0:
			#convert coordinate index into processor rank 
			#print("---rank = {}, sending first row to {}, vec of size = {}".format(world.rank,world.BottomNeighRank, len(vec[0:world.numPointsX])))
			mpi4py.MPI.COMM_WORLD.Send(vec[0:world.numPointsX], dest=world.BottomNeighRank, tag=24)
	
	elif neigh == "bottom":

		#2. sent bottom data, j-1 index
		#this will be the u_{i,j-1} data
		#last row of data processor has acces too 
		
		if world.procYID < world.procs_per_yaxis -1:
			#print("--rank = {}, sending last row to {}".format(world.rank, world.TopNeighRank))

			startIdxrRow = int(world.numPointsX*(world.numPointsY-1))
			endIdxRow    = int(world.numPointsX*world.numPointsY)

			mpi4py.MPI.COMM_WORLD.Send(vec[startIdxrRow:endIdxRow], dest=world.TopNeighRank, tag=25)	

	elif neigh == "right":
		#3. send data to right neighbor
		#this will be u_{i-1,j} data because x increases -->
		if world.procXID < world.procs_per_xaxis -1: 

			#print("rank {} sending right colum to rank {}".format(world.rank, world.RightNeighRank))
			#note, data is not contiguous, need a buffer
			buffer = np.zeros(world.numPointsY)
			for i in range(0, world.numPointsY):
				indx = (world.numPointsX*(i+1)) - 1
				buffer[i] = vec[indx]

			#now send buffer data 
			mpi4py.MPI.COMM_WORLD.Send(buffer, dest=world.RightNeighRank, tag = 32)

	else :
		#4. send data to left neighboor	
		#this will be u_{i+1,j} data, because x+ -->
		if world.procXID > 0: 

			#print("rank {} sending left column to rank {}".format(world.rank, world.LeftNeighRank))
			#note, data is not contiguous because it's stored by rowss
			buffer = np.zeros(world.numPointsY)

			for i in range(0, world.numPointsY):
				indx = world.numPointsX*i
				buffer[i] = vec[indx]

			#now send buffer data to previous processor
			mpi4py.MPI.COMM_WORLD.Send(buffer, dest = world.LeftNeighRank, tag = 34)
			



#function to recieve neighboor data
#local communcation
#refering to indices (i,j) (x,y)

def recieveData(neigh, world, dataArray):

	if neigh == "top":
	
		#1. recieve top data, index j+1
		#recieve from top processor (though this is physically down)
		if world.procYID < world.procs_per_yaxis-1:
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.TopNeighRank, tag=24)
	
	elif neigh == "bottom":

		#2. recieve bottom data, index j-1
		if world.procYID > 0:
			#print("recieving data for rank = {} from rank = {}".format(world.rank, world.BottomNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.BottomNeighRank, tag=25)

	elif neigh == "right":

		#3. recieve left data, u_{i-1,j}
		if world.procXID > 0:
			#print("recieviing left data for rank {} from rank {}".format(world.rank, world.LeftNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.LeftNeighRank, tag=32)
	

	else:

		#4. recieve right data, u_{i+1,j}
		if world.procXID < world.procs_per_xaxis -1:
			#print("recieving right data for rank {} from rank {}".format(world.rank, world.RightNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.RightNeighRank, tag=34)	


#functions for communication
def sendPerData(vec, neigh, world):

	if neigh == "top":

		#1. send top data, j+1 index
		#if the rank is >0, send top data to previous rank
		#this will be the u_{i,j+1} data 
		#its the first vector the processor has access to

		if world.procYID == 0:
			#convert coordinate index into processor rank 
			#print("---rank = {}, sending first row to {}, vec of size = {}".format(world.rank,world.BottomNeighRank, len(vec[0:world.numPointsX])))
			mpi4py.MPI.COMM_WORLD.Send(vec[0:world.numPointsX], dest=world.BottomNeighRank, tag=24)
	
	elif neigh == "bottom":

		#2. sent bottom data, j-1 index
		#this will be the u_{i,j-1} data
		#last row of data processor has acces too 
		
		if world.procYID == world.procs_per_yaxis -1:
			#print("--rank = {}, sending last row to {}".format(world.rank, world.TopNeighRank))

			startIdxrRow = int(world.numPointsX*(world.numPointsY-1))
			endIdxRow    = int(world.numPointsX*world.numPointsY)

			mpi4py.MPI.COMM_WORLD.Send(vec[startIdxrRow:endIdxRow], dest=world.TopNeighRank, tag=25)	

	elif neigh == "right":
		#3. send data to right neighbor
		#this will be u_{i-1,j} data because x increases -->
		if world.procXID == world.procs_per_xaxis -1: 

			#print("rank {} sending right colum to rank {}".format(world.rank, world.RightNeighRank))
			#note, data is not contiguous, need a buffer
			buffer = np.zeros(world.numPointsY)
			for i in range(0, world.numPointsY):
				indx = (world.numPointsX*(i+1)) - 1
				buffer[i] = vec[indx]

			#now send buffer data 
			mpi4py.MPI.COMM_WORLD.Send(buffer, dest=world.RightNeighRank, tag = 32)

	else :
		#4. send data to left neighboor	
		#this will be u_{i+1,j} data, because x+ -->
		if world.procXID == 0: 

			#print("rank {} sending left column to rank {}".format(world.rank, world.LeftNeighRank))
			#note, data is not contiguous because it's stored by rowss
			buffer = np.zeros(world.numPointsY)

			for i in range(0, world.numPointsY):
				indx = world.numPointsX*i
				buffer[i] = vec[indx]

			#now send buffer data to previous processor
			mpi4py.MPI.COMM_WORLD.Send(buffer, dest = world.LeftNeighRank, tag = 34)
			



#function to recieve neighboor data
#local communcation
#refering to indices (i,j) (x,y)

def recievePerData(neigh, world, dataArray):

	if neigh == "top":
	
		#1. recieve top data, index j+1
		#recieve from top processor (though this is physically down)
		if world.procYID == world.procs_per_yaxis-1:
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.TopNeighRank, tag=24)
	
	elif neigh == "bottom":

		#2. recieve bottom data, index j-1
		if world.procYID == 0:
			#print("recieving data for rank = {} from rank = {}".format(world.rank, world.BottomNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.BottomNeighRank, tag=25)

	elif neigh == "right":

		#3. recieve left data, u_{i-1,j}
		if world.procXID == 0:
			#print("recieviing left data for rank {} from rank {}".format(world.rank, world.LeftNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.LeftNeighRank, tag=32)
	

	else:

		#4. recieve right data, u_{i+1,j}
		if world.procXID == world.procs_per_xaxis -1:
			#print("recieving right data for rank {} from rank {}".format(world.rank, world.RightNeighRank))
			mpi4py.MPI.COMM_WORLD.Recv(dataArray, source=world.RightNeighRank, tag=34)	



"""
function for getting neighboor data. Used for both laplacian
and advection operators. Only need to call once for these 
examples. 

"""
def getNeighData(vec, world, topData, bottomData, leftData, rightData):

	#sending top neighboor data
	sendData(vec, "top", world)
	recieveData("top", world, topData)

	#sending bottom neighboor data
	sendData(vec, "bottom", world)
	recieveData("bottom", world, bottomData)

	#sending right neighboor data
	sendData(vec, "right", world)
	recieveData("right", world, leftData)

	#sending left neighoor data
	sendData(vec, "left", world)
	recieveData("left", world, rightData)

	"""
	print("(procXID, pricYID)  = ({}, {}), leftData = {}".format(world.procXID, world.procYID, leftData))
	print("(procXID, pricYID)  = ({}, {}), RightData = {}".format(world.procXID, world.procYID, rightData))

	print("(procXID, pricYID)  = ({}, {}), bottomData = {}".format(world.procXID, world.procYID, bottomData))
	print("(procXID, pricYID)  = ({}, {}), TopData = {}".format(world.procXID, world.procYID, topData))
	"""


	return topData, bottomData, leftData, rightData


def getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData):


	#sending top neighboor data
	sendPerData(vec, "top", world)
	recievePerData("top", world, topData)

	#sending bottom neighboor data
	sendPerData(vec, "bottom", world)
	recievePerData("bottom", world, bottomData)

	#sending right neighboor data
	sendPerData(vec, "right", world)
	recievePerData("right", world, leftData)

	#sending left neighoor data
	sendPerData(vec, "left", world)
	recievePerData("left", world, rightData)

	return topData, bottomData, leftData, rightData



def laplacianDirichlet(vec, epsilon, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

	#2. matrix free application of matrix times vector 
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data 

			indx = j*world.numPointsX + i 
			current_val = -4.0*vec[indx]

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += vec[indx+1]

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += vec[indx+1] + LeftNeighData[j]

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += vec[indx-1]

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val += vec[indx-1] + RightNeighData[j]

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += vec[indx-1] + vec[indx+1]

			#top and bottom bcs 
			if j == 0: 	#bottom data, get u_{k,j-1}

				#if rank == 0, need bottom boundary condition
				if world.procYID == 0:
					#print("j = {}, i = {}, indx = {}, indx + pointsXnoBC = {}".format(j, i, indx, indx + world.numPointsXnoBC))
					#current_val += world.BottomBoundaryCond[i] + vec[indx + world.numPointsXnoBC]
					current_val += vec[indx + world.numPointsX]

				#if not, use data from neighbooring processor
				else:
					current_val += BottomNeighData[i] + vec[indx + world.numPointsX]

			elif j == world.numPointsY -1: 	#top data, get u_{k,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1 ):	
					#current_val += world.TopBoundaryCond[i] + vec[indx - world.numPointsXnoBC]
					current_val += vec[indx - world.numPointsX]

				#else, use data from neighboring processors
				else:
					current_val += TopNeighData[i] + vec[indx - world.numPointsX]

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += vec[indx - world.numPointsX] + vec[indx + world.numPointsX]	

			#now scale by h^2 and save value in ouy
			out[indx] = epsilon*( current_val / world.dx**2 )


	return out


"""
laplacian operator with homogeneous Neumann boundary conditions

"""
def laplacianNeumann(vec, epsilon, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:

		#if world.IamRoot: print("---calling regular data---")

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

	#2. matrix free application of matrix times vector 
	#print("---compute stencil---")
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data 

			indx = j*world.numPointsX + i 
			current_val = -4.0*vec[indx]

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					#print("inside left-x boundary condition for 1p")
					current_val += (2.0/3.0)*vec[indx+1] + (4.0/3.0)*vec[indx]

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += vec[indx+1] + LeftNeighData[j]

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					#print("inside right-x boundary condition for 1p")
					current_val += (2.0/3.0)*vec[indx-1] + (4.0/3.0)*vec[indx]

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					#print("inside else for right-x for 1p")
					current_val += vec[indx-1] + RightNeighData[j]

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += vec[indx-1] + vec[indx+1]

			#top and bottom bcs 
			if j == 0: 	#bottom data, get u_{k,j-1}

				#if rank == 0, need bottom boundary condition
				if world.procYID == 0:
					#print("inside bottom-y boundary condition for 1p")
					current_val += (2.0/3.0)*vec[indx + world.numPointsX] + (4.0/3.0)*vec[indx]

				#if not, use data from neighbooring processor
				else:
					current_val += BottomNeighData[i] + vec[indx + world.numPointsX]

			elif j == world.numPointsY -1: 	#top data, get u_{k,j+1}			
				
				if (world.procYID == world.procs_per_yaxis-1):	
					#print("inside top-y boundary condition for 1p")
					current_val += (2.0/3.0)*vec[indx - world.numPointsX] + (4.0/3.0)*vec[indx]

				#else, use data from neighboring processors
				else:
					#print("inside else for top-y boundary for 1p")
					current_val += TopNeighData[i] + vec[indx - world.numPointsX]

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += vec[indx - world.numPointsX] + vec[indx + world.numPointsX]	

			#now scale by h^2 and save value in ouy
			out[indx] = epsilon*( current_val / world.dx**2 )


	return out 

"""
non-linear laplacian operator with homogeneous Neumann boundary conditions
This is the Jacobian of the d^2 u^2 /dx term for Porous medium equation

parameters:
#	1. vec: vector multiplying operator
#	2. u  : vector used in the nonlinear Jacobian, what j depends on
#	3. epsilion: used to scale laplacian
#	4. dt : timestep, multiply output as well
#	5. world: class to access info such as arrays for data, grid points etc
"""
def nonLinLapPeriodic(vec, u, epsilon, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	jactopData = np.zeros(world.numPointsX)
	jacbottomData = np.zeros(world.numPointsX)
	jacleftData = np.zeros(world.numPointsX)
	jacrightData = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

		[JacUTopData, JacUBotData, JacULeftData, JacURightData] = getNeighData(u, world, jactopData, jacbottomData, jacleftData, jacrightData)

		[JacUTopData, JacUBotData, JacULeftData, JacURightData] = getExtraPeriodicData(u, world, jactopData, jacbottomData, jacleftData, jacrightData)


	#2. matrix free application of matrix times vector 
	N = world.numPointsX
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data 

			indx = j*world.numPointsX + i 
			current_val = -4.0*vec[indx]*u[indx]

			#left and right neighboor data
			if i == 0: #left data

				current_val += vec[indx+1]*u[indx+1] + LeftNeighData[j]*JacULeftData[j]

			elif i == world.numPointsY -1:  #right data	

				current_val += vec[indx-1]*u[indx-1] + RightNeighData[j]*JacURightData[j]

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += vec[indx-1]*u[indx-1] + vec[indx+1]*u[indx+1]

			#top and bottom bcs 
			if j == 0: 	#bottom data, get u_{k,j-1}

				current_val += BottomNeighData[i]*JacUBotData[i] + vec[indx + world.numPointsX]*u[indx+N]

			elif j == world.numPointsY -1: 	#top data, get u_{k,j+1}			
				
				current_val += TopNeighData[i]*JacUTopData[i] + vec[indx - world.numPointsX]*u[indx-N]

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += vec[indx - N]*u[indx-N] + vec[indx + N]*u[indx+N]	

			#now scale by h^2 and save value in ouy
			out[indx] = 2.0*epsilon*( current_val / world.dx**2 )


	return out


"""
laplacian operator with periodic boundary conditions

"""
def laplacianPeriodic(vec, epsilon, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

	#2. matrix free application of matrix times vector 
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data 

			indx = j*world.numPointsX + i 
			current_val = -4.0*vec[indx]

			#left and right neighboor data
			if i == 0: #left data

				current_val += vec[indx+1] + LeftNeighData[j]

			elif i == world.numPointsY -1:  #right data	

				current_val += vec[indx-1] + RightNeighData[j]

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += vec[indx-1] + vec[indx+1]

			#top and bottom bcs 
			if j == 0: 	#bottom data, get u_{k,j-1}

				current_val += BottomNeighData[i] + vec[indx + world.numPointsX]

			elif j == world.numPointsY -1: 	#top data, get u_{k,j+1}			
				
				current_val += TopNeighData[i] + vec[indx - world.numPointsX]

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += vec[indx - world.numPointsX] + vec[indx + world.numPointsX]	

			#now scale by h^2 and save value in ouy
			out[indx] = epsilon*( current_val / world.dx**2 )


	return out 

#laplacian of u^2 for porous medium equation
#this will be used for the rhs of the porous medium equation
def laplacianUsquared(vec, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData    = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData   = np.zeros(world.numPointsX)
	rightData  = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

	#2. compute stencil
	N = world.numPointsX

	for j in range(N):
		for i in range(N):

			indx = j*world.numPointsX + i
			current_val = -4*vec[indx]**2

			#left and right neighboor data
			if i == 0: #left data

				current_val += vec[indx+1]**2 + LeftNeighData[j]**2

			elif i == world.numPointsY -1:  #right data	

				current_val += vec[indx-1]**2 + RightNeighData[j]**2

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += vec[indx-1]**2 + vec[indx+1]**2

			#top and bottom bcs 
			if j == 0: 	#bottom data, get u_{k,j-1}

				current_val += BottomNeighData[i]**2 + vec[indx + N]**2

			elif j == world.numPointsY -1: 	#top data, get u_{k,j+1}			
				
				current_val += TopNeighData[i]**2 + vec[indx - N]**2

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += vec[indx - N]**2 + vec[indx + N]**2	

			#now scale by h^2 and save value in ouy
			out[indx] = ( current_val / world.dx**2 )

	return out 



#advection operator with homogeneous dirichlet boundary conditions
def advectionDirichlet(vec, alpha, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += vec[indx+1]

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += ( vec[indx+1] - LeftNeighData[j] )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += -vec[indx-1]

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j] - vec[indx-1] )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1] - vec[indx-1])


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += vec[indx + world.numPointsX]

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX] - BottomNeighData[i])

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1 and world.size-1 != 0):	
					#current_val += world.TopBoundaryCond[i] + vec[indx - world.numPointsXnoBC]
					current_val +=  -vec[indx - world.numPointsX]

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i] - vec[indx - world.numPointsX])

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX] - vec[indx - world.numPointsX])	

			#3. scale the solution 
			out[indx] = alpha*(current_val / (2.0 * world.dx) )
					


	return out 



#advection with Neumann boundary conditions
def advectionNeumann(vec, alpha, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	#to avoid possible communication locks, onl
	#call if we more than 1 processor
	if world.size > 1:

		#if world.IamRoot: print("---calling regular data---")

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)


	#2. matrix free application of matrix times vector 
	#operator is second order fd
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += 4.0/3.0 *(vec[indx+1] - vec[indx] )

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += ( vec[indx+1] - LeftNeighData[j] )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += 4.0/3.0 *(vec[indx] - vec[indx-1] )

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j] - vec[indx-1] )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1] - vec[indx-1])


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += 4.0/3.0*(vec[indx + world.numPointsX] - vec[indx] )

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX] - BottomNeighData[i])

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1):	

					current_val +=  4.0/3.0 *(vec[indx]- vec[indx - world.numPointsX])

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i] - vec[indx - world.numPointsX])

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX] - vec[indx - world.numPointsX])	

			#3. scale the solution 
			out[indx] = alpha*(current_val / (2.0 * world.dx) )
				

	return out 

#advection with periodic boundary conditions
def advectionPeriodic(vec, alpha, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:
		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

	N = world.numPointsX

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	#print("---computing stencil---")
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					#print("indx = {}, indx+ (N-1) = {}, vec[indx + N -1] = {}".format(indx, indx + (N-1), vec[indx + (N-1)] ))
					current_val += (vec[indx+1] - LeftNeighData[j] )

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += (vec[indx+1] - LeftNeighData[j] )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += (RightNeighData[j] - vec[indx-1] )

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j] - vec[indx-1] )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1] - vec[indx-1])


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += (vec[indx + world.numPointsX] - BottomNeighData[i] )

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX] - BottomNeighData[i])

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1):	

					current_val +=  (TopNeighData[i]- vec[indx - world.numPointsX])

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i] - vec[indx - world.numPointsX])

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX] - vec[indx - world.numPointsX])	

			#3. scale the solution 
			out[indx] = alpha*(current_val / (2.0 * world.dx) )
				

	return out 

#nonlinear advection for JTV of Burger's equations
#NOTE: this is with Dirichlet bc's 
def nonLinAdvecDirichlet(vec, u, alpha, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData    = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData   = np.zeros(world.numPointsX)
	rightData  = np.zeros(world.numPointsX)

	jactopData    = np.zeros(world.numPointsX)
	jacbottomData = np.zeros(world.numPointsX)
	jacleftData   = np.zeros(world.numPointsX)
	jacrightData  = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[JacUTopData, JacUBotData, JacULeftData, JacURightData] = getNeighData(u, world, jactopData, jacbottomData, jacleftData, jacrightData)

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += vec[indx+1]*u[indx+1]

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += ( vec[indx+1]*u[indx+1] - LeftNeighData[j]*JacULeftData[j] )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += -vec[indx-1]*u[indx-1]

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j]*JacURightData[j] - vec[indx-1]*u[indx-1] )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1]*u[indx+1] - vec[indx-1]*u[indx-1])


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += vec[indx + world.numPointsX]*u[indx + world.numPointsX]

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX]*u[indx + world.numPointsX] - BottomNeighData[i]*JacUBotData[i])

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1 and world.size-1 != 0):	
					#current_val += world.TopBoundaryCond[i] + vec[indx - world.numPointsXnoBC]
					current_val +=  -vec[indx - world.numPointsX]*u[indx-world.numPointsX]

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i]*JacUTopData[i] - vec[indx - world.numPointsX]*u[indx-world.numPointsX])

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX]*u[indx+world.numPointsX] - vec[indx - world.numPointsX]*u[indx-world.numPointsX])	

			#3. scale the solution 
			out[indx] = alpha*(current_val / (2.0 * world.dx) )
					
	return out 



"""
nonlinear advection with periodic boundary conditions
this is used for Burger's equation
parameters:
#	1. vec: vector multiplying operator
#	2. u  : vector used in the nonlinear Jacobian, what j depends on
#	3. epsilion: used to scale laplacian
#	4. dt : timestep, multiply output as well
#	5. world: class to access info such as arrays for data, grid points etc
"""
def nonLinAdvecPeriodic(vec, u, alpha, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	jactopData = np.zeros(world.numPointsX)
	jacbottomData = np.zeros(world.numPointsX)
	jacleftData = np.zeros(world.numPointsX)
	jacrightData = np.zeros(world.numPointsX)


	if world.size > 1:

		#print("---calling regular data---")
		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		#print("---calling extra periodic data---")
		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

		#print("---calling regular data---")
		[JacUTopData, JacUBotData, JacULeftData, JacURightData] = getNeighData(u, world, jactopData, jacbottomData, jacleftData, jacrightData)

		#print("---calling extra periodic data---")
		[JacUTopData, JacUBotData, JacULeftData, JacURightData] = getExtraPeriodicData(u, world, jactopData, jacbottomData, jacleftData, jacrightData)


	N = world.numPointsX

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	#print("---computing stencil---")
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					#print("indx = {}, indx+ (N-1) = {}, vec[indx + N -1] = {}".format(indx, indx + (N-1), vec[indx + (N-1)] ))
					current_val += (vec[indx+1]*u[indx+1] - LeftNeighData[j]*JacULeftData[j] )

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += ( vec[indx+1]*u[indx+1] - LeftNeighData[j]*JacULeftData[j] )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += (RightNeighData[j]*JacURightData[j] - vec[indx-1]*u[indx-1] )

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j]*JacURightData[j] - vec[indx-1]*u[indx-1] )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1]*u[indx+1] - vec[indx-1]*u[indx-1])


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += (vec[indx + world.numPointsX]*u[indx+N] - BottomNeighData[i]*JacUBotData[i] )

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX]*u[indx+N] - BottomNeighData[i]*JacUBotData[i])

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1):	

					current_val +=  (TopNeighData[i]*JacUTopData[i]- vec[indx - world.numPointsX]*u[indx-N])

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i]*JacUTopData[i] - vec[indx - world.numPointsX]*u[indx-N])

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX]*u[indx+N] - vec[indx - world.numPointsX]*u[indx-N])	

			#3. scale the solution 
			out[indx] = alpha*(current_val / (2.0 * world.dx) )
				

	return out 

#advection of u^2, this is for the rhs of burger's equation
#note this is with Dirichlet boundary conditions
def advectionUsquaredDir(vec, world):
	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += vec[indx+1]**2

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += ( vec[indx+1]**2 - LeftNeighData[j]**2 )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += -vec[indx-1]**2

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j]**2 - vec[indx-1]**2 )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1]**2 - vec[indx-1]**2)


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += vec[indx + world.numPointsX]**2

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + world.numPointsX]**2 - BottomNeighData[i]**2)

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1 and world.size-1 != 0):	
					#current_val += world.TopBoundaryCond[i] + vec[indx - world.numPointsXnoBC]
					current_val +=  -vec[indx - world.numPointsX]**2

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i]**2 - vec[indx - world.numPointsX]**2)

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + world.numPointsX]**2 - vec[indx - world.numPointsX]**2)	

			#3. scale the solution 
			out[indx] = (current_val / (4.0 * world.dx) )
					


	return out 

#advection of u^2 with periodic boundary conditions
#this function will be used for the rhs of Burger's equation 
#note it's 0.5 (u^2)_x, so he centered differnce stencil is divided
#by 4h instead of 2h
def advectionUsquared(vec, world):

	#1. set up out vector
	out = np.zeros_like(vec)

	topData = np.zeros(world.numPointsX)
	bottomData = np.zeros(world.numPointsX)
	leftData = np.zeros(world.numPointsX)
	rightData = np.zeros(world.numPointsX)

	if world.size > 1:
		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getNeighData(vec, world, topData, bottomData, leftData, rightData)

		[TopNeighData, BottomNeighData, LeftNeighData, RightNeighData] = getExtraPeriodicData(vec, world, topData, bottomData, leftData, rightData)

	N = world.numPointsX

	#2. matrix free application of matrix times vector 
	#operator is second order fd
	#print("---computing stencil---")
	for j in range(0, world.numPointsY): #depth of processor

		for i in range(0, world.numPointsX):	#length of data

			indx = j*world.numPointsX + i
			current_val = 0.0 

			#left and right neighboor data
			if i == 0: #left data

				#if procXID == 0, then the data u_{i-1,j} dne
				if world.procXID == 0:
					current_val += (vec[indx+1]**2 - LeftNeighData[j]**2 )

				#if not xid = 0, then use the left neighboor data
				else:
					current_val += (vec[indx+1]**2 - LeftNeighData[j]**2 )

			elif i == world.numPointsY -1:  #right data	

				#if at the last element and at the boundary, then 
				#u_{i+1,j} dne
				if world.procXID == world.procs_per_xaxis -1: 
					current_val += (RightNeighData[j]**2 - vec[indx-1]**2 )

				#if at last point in processor domain but not at 
				#end of physical domain, use right neighboor data
				#for u_{i+1,j}
				else: 	
					current_val +=  (RightNeighData[j]**2 - vec[indx-1]**2 )

			#if inbetween points of processor, we have acess to left and right points
			# val += u_{i+1,j} + u_{i-1,j}
			else: 	
				current_val += (vec[indx+1]**2 - vec[indx-1]**2)


			if j == 0: 

				#if the procYID is 0, then u_{i, j-1} dne
				if world.procYID == 0: 
					current_val += (vec[indx + N]**2 - BottomNeighData[i]**2 )

				#if not, use data from neighbooring processor
				else:
					current_val +=  (vec[indx + N]**2 - BottomNeighData[i]**2)

			elif j == world.numPointsY -1: 	#top data, get u_{i,j+1}			
				
				#if rank == size-1, then need top boundary condition
				#if world.rank == world.size-1:
				if (world.procYID == world.procs_per_yaxis-1):	

					current_val +=  (TopNeighData[i]**2- vec[indx - N]**2 )

				#else, use data from neighboring processors
				else:
					current_val += (TopNeighData[i]**2 - vec[indx - N]**2 )

			else: 	#have acces to data within processors
				#print("j = {}, i = {}, indx = {}".format(j, i, indx))
				current_val += (vec[indx + N]**2 - vec[indx - N]**2)	

			#3. scale the solution 
			out[indx] = (current_val / (4.0 * world.dx) )
				

	return out 

