import mpi4py.MPI

def create_ptopo():
   size = mpi4py.MPI.COMM_WORLD.Get_size()
   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   if size % 6 != 0:
      raise Exception('Number of processes should be a multiple of 6 ...')

   #      +---+
   #      | 4 |
   #  +---+---+---+---+
   #  | 3 | 0 | 1 | 2 |
   #  +---+---+---+---+
   #      | 5 |
   #      +---+

   # Distributed Graph 
   # N, S, W, E ordering
   if rank == 0:
      sources = [4, 5, 3, 1]
      destinations = sources
   if rank == 1:
      sources = [4, 5, 0, 2]
      destinations = sources
   if rank == 2:
      sources = [4, 5, 1, 3]
      destinations = sources
   if rank == 3:
      sources = [4, 5, 0, 2]
      destinations = sources
   if rank == 4:
      sources = [2, 0, 3, 1]
      destinations = sources
   if rank == 5:
      sources = [0, 2, 3, 1]
      destinations = sources
   
   return mpi4py.MPI.COMM_WORLD.Create_dist_graph_adjacent(sources, destinations), rank
