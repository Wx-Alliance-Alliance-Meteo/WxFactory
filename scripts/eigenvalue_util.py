from   itertools import product
import os
from   typing    import List, Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray
import scipy.sparse
from scipy.sparse import csc_matrix

try:
   from tqdm import tqdm
except ModuleNotFoundError:
   tqdm_message_printed = MPI.COMM_WORLD.rank != 0
   def tqdm(a):
      global tqdm_message_printed
      if not tqdm_message_printed:
         print(f'Module "tqdm" was not found. You need it if you want to see progress bars')
         tqdm_message_printed = True
      return a

from solvers import MatvecOp

def gen_matrix(matvec: MatvecOp,
               jac_file_name: Optional[str] = None,
               compressed: Optional[bool] = None,
               local: bool = False) \
                  -> Optional[scipy.sparse.csc_matrix]:
   """
   Compute and store the Jacobian matrix. It may be computed either as a full or sparse matrix
   (faster as full, but it may take a *lot* of memory for large matrices). Always stored as
   sparse.
   :param matvec: Operator to compute the action of the jacobian on a vector. Holds vector shape and variable type
   :param jac_file_name: If present, path to the file where the jacobian will be stored
   """

   # neq, ni, nj = matvec.shape
   n_loc = matvec.size

   rank = MPI.COMM_WORLD.Get_rank()
   size = MPI.COMM_WORLD.Get_size()

   if compressed is None:
      compressed = n_loc * size > 150000

   if rank == 0:
      print(f'Generating jacobian matrix. Shape {matvec.shape}')

   # Global unit vector we will multiply the matrix with
   # (Section that corresponds to this tile)
   Qid = numpy.zeros((n_loc), dtype=matvec.dtype)

   def progress(a): return a
   if rank == 0:
      progress = tqdm

   # Compute the matrix one column at a time by multiplying by a basis vector
   idx = 0
   indices = list(range(n_loc))
   columns: List[NDArray | csc_matrix | None] = [None for _ in range(len(indices * size))]
   for r in range(size):
      if rank == 0: print(f'Tile {r+1}/{size}')
      for i in progress(indices):
         if rank == r: Qid[i] = 1.0
         col = matvec(Qid.flatten())
         ccol = csc_matrix(col.reshape((col.size, 1))) if compressed else col
         columns[idx] = ccol
         idx += 1
         Qid[i] = 0.0

   J_tile = scipy.sparse.hstack(columns, format='csc')

   # Gather the matrix segments from all PEs and save the matrix to file
   if local:
      J_square_tile = J_tile[:, n_loc * rank : n_loc * (rank + 1)]

      if jac_file_name is not None:
         print(f'Saving jacobian to {os.path.relpath(jac_file_name)}')
         scipy.sparse.save_npz(jac_file_name, J_square_tile)

      return J_square_tile

   # We want a global matrix. Gather the tiles into a single matrix
   J_tile_list = MPI.COMM_WORLD.gather(J_tile, root=0)
   if rank == 0:
      print('')

      J_full = scipy.sparse.vstack(J_tile_list)

      if jac_file_name is not None:
         print(f'Saving jacobian to {os.path.relpath(jac_file_name)}')
         scipy.sparse.save_npz(jac_file_name, J_full)

      return J_full

   return None
