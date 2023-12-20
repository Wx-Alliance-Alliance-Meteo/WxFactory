from   itertools import product
import os
from   typing    import Optional

from mpi4py import MPI
import numpy
import scipy.sparse

try:
   from tqdm import tqdm
except ModuleNotFoundError:
   print(f'Module "tqdm" was not found. You need it if you want to see progress bars')
   def tqdm(a): return a

from solvers                import MatvecOp

def gen_matrix(matvec: MatvecOp, jac_file_name: Optional[str] = None, compressed: Optional[bool] = None, permute: bool = False) -> Optional[scipy.sparse.csc_matrix]:
   """
   Compute and store the Jacobian matrix. It may be computed either as a full or sparse matrix
   (faster as full, but it may take a *lot* of memory for large matrices). Always stored as
   sparse.
   :param matvec: Operator to compute the action of the jacobian on a vector. Holds vector shape and variable type
   :param jac_file_name: If present, path to the file where the jacobian will be stored
   :param permute: Whether to permute matrix rows and columns to groups entries associated with an element into a block
   """

   neq, ni, nj = matvec.shape
   n_loc = matvec.size

   rank = MPI.COMM_WORLD.Get_rank()
   size = MPI.COMM_WORLD.Get_size()

   if compressed is None:
      compressed = n_loc * size > 150000

   print(f'Generating jacobian matrix. Shape {matvec.shape}')

   Qid = numpy.zeros(matvec.shape, dtype=matvec.dtype)
   # J = scipy.sparse.csc_matrix((n_loc, size*n_loc), dtype=matvec.dtype) if compressed else numpy.zeros((n_loc, size*n_loc))

   def progress(a): return a
   if rank == 0:
      progress = tqdm

   # Compute the matrix one column at a time by multiplying by a basis vector
   idx = 0
   indices = [i for i in product(range(neq), range(ni), range(nj))]
   columns = [None for _ in range(len(indices))]
   for r in range(size):
      if rank == 0: print(f'Tile {r+1}/{size}')
      for (i, j, k) in progress(indices):
         if rank == r: Qid[i, j, k] = 1.0
         col = matvec(Qid.flatten())
         ccol = scipy.sparse.csc_matrix(col.reshape((col.size, 1))) if compressed else col
         columns[idx] = ccol
         # J[:, idx] = ccol
         idx += 1
         Qid[i, j, k] = 0.0

   J = scipy.sparse.hstack(columns)

   # If it wasn't already compressed, do it now
   if not compressed: J = scipy.sparse.csc_matrix(J)

   # Gather the matrix segments from all PEs and save the matrix to file
   J_comm   = MPI.COMM_WORLD.gather(J, root=0)
   if rank == 0:
      print('')

      glb_J = scipy.sparse.vstack(J_comm)

      if permute:
         print(f'Permute was not implemented for cartesian grids and (possibly) 3D problems')
         # p = permutations()
         # glb_J = csc_matrix(lil_matrix(glb_J)[p, :][:, p])

      if jac_file_name is not None:
         print(f'Saving jacobian to {os.path.relpath(jac_file_name)}')
         scipy.sparse.save_npz(jac_file_name, glb_J)

      return glb_J
   else:
      return None
