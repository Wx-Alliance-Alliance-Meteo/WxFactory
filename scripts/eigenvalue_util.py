from   itertools import product
import os
import pickle
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

from common.device import Device, default_device
from solvers import MatvecOp

def gen_matrix(matvec: MatvecOp,
               jac_file_name: Optional[str] = None,
               compressed: Optional[bool] = None,
               local: bool = False,
               device: Device = default_device) \
                  -> Optional[scipy.sparse.csc_matrix]:
   """
   Compute and store the Jacobian matrix. It may be computed either as a full or sparse matrix
   (faster as full, but it may take a *lot* of memory for large matrices). Always stored as
   sparse.
   :param matvec: Operator to compute the action of the jacobian on a vector. Holds vector shape and variable type
   :param jac_file_name: If present, path to the file where the jacobian will be stored
   """

   xp = device.xp

   # neq, ni, nj = matvec.shape
   n_loc = matvec.size

   rank = MPI.COMM_WORLD.Get_rank()
   size = MPI.COMM_WORLD.Get_size()

   if compressed is None:
      # compressed = n_loc * size > 150000
      compressed = True

   if rank == 0:
      print(f'Generating jacobian matrix. Shape {matvec.shape}')

   # Global unit vector we will multiply the matrix with
   # (Section that corresponds to this tile)
   Qid = xp.zeros((n_loc), dtype=matvec.dtype)

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
         col = device.to_host(matvec(Qid.flatten()))
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


def save_matrices(file, matrices):
   arrays_dict = {}
   num_mat = 0
   for matrix in matrices:
      if matrix.format in ('csc', 'csr', 'bsr'):
         arrays_dict[f'indices{num_mat}'] = matrix.indices
         arrays_dict[f'indptr{num_mat}']  = matrix.indptr
      elif matrix.format == 'dia':
         arrays_dict[f'offsets{num_mat}'] = matrix.offsets
      elif matrix.format == 'coo':
         arrays_dict[f'row{num_mat}'] = matrix.row,
         arrays_dict[f'col{num_mat}'] = matrix.col
      else:
         raise NotImplementedError(f'Save is not implemented for sparse matrix of format {matrix.format}.')

      arrays_dict[f'format{num_mat}'] = matrix.format.encode('ascii')
      arrays_dict[f'shape{num_mat}']  = matrix.shape
      arrays_dict[f'data{num_mat}']   = matrix.data

      num_mat += 1

   numpy.savez(file, **arrays_dict)

PICKLE_KWARGS = dict(allow_pickle=False)

def load_matrices(file):
   loaded = numpy.load(file, **PICKLE_KWARGS)

   if 'format0' not in loaded:
      raise ValueError(f'The file {file} does not contain a sparse matrix.')

   num_loaded = 0
   matrices = []
   while True:
      try:
         matrix_format = loaded[f'format{num_loaded}'].item()
      except KeyError:
         break

      if not isinstance(matrix_format, str):
         # Play safe with Python 2 vs 3 backward compatibility;
         # files saved with SciPy < 1.0.0 may contain unicode or bytes.
         matrix_format = matrix_format.decode('ascii')

      try:
         cls = getattr(scipy.sparse, f'{matrix_format}_matrix')
      except AttributeError as e:
         raise ValueError(f'Unknown matrix format "{matrix_format}"') from e

      if matrix_format in ('csc', 'csr', 'bsr'):
         matrices.append(cls((loaded[f'data{num_loaded}'],
                     loaded[f'indices{num_loaded}'],
                     loaded[f'indptr{num_loaded}']),
                     shape=loaded[f'shape{num_loaded}']))
      elif matrix_format == 'dia':
         matrices.append(cls((loaded[f'data{num_loaded}'],
                     loaded[f'offsets{num_loaded}']),
                     shape=loaded[f'shape{num_loaded}']))
      elif matrix_format == 'coo':
         matrices.append(cls((loaded[f'data{num_loaded}'],
                     (loaded[f'row{num_loaded}'], loaded[f'col{num_loaded}'])),
                     shape=loaded[f'shape{num_loaded}']))
      else:
         raise NotImplementedError(f'Load is not implemented for sparse matrix of format {matrix_format}.')

      num_loaded += 1

   return matrices

ignore_list = ['param', 'permutation', 'precond', 'precond_p']
def store_matrix_set(filename, matrix_set):
   with open(filename, 'wb') as output_file:
      pickle.dump(matrix_set['param'], output_file)
      pickle.dump(matrix_set['permutation'], output_file)
      key_list = list(matrix_set)
      pickle.dump(key_list, output_file)
      all_matrices = []
      for key in key_list:
         if key in ignore_list: continue
         entry = matrix_set[key]
         numpy.save(output_file, entry['rhs'])
         numpy.save(output_file, entry['rhs_p'])
         all_matrices.append(entry['matrix'])
         all_matrices.append(entry['matrix_p'])

      if 'precond' in key_list:
         all_matrices.append(matrix_set['precond'])
         all_matrices.append(matrix_set['precond_p'])

      save_matrices(output_file, all_matrices)

   load_matrix_set(filename)

def load_matrix_set(filename):
   matrix_set = {}
   with open(filename, 'rb') as input_file:
      matrix_set['param'] = pickle.load(input_file)
      matrix_set['permutation'] = pickle.load(input_file)
      key_list = pickle.load(input_file)
      for key in key_list:
         if key in ignore_list: continue
         matrix_set[key] = {}
         matrix_set[key]['rhs'] = numpy.load(input_file)
         matrix_set[key]['rhs_p'] = numpy.load(input_file)

      all_matrices = load_matrices(input_file)
      num_mat = 0
      for key in key_list:
         if key in ignore_list: continue
         matrix_set[key]['matrix'] = all_matrices[num_mat]
         matrix_set[key]['matrix_p'] = all_matrices[num_mat + 1]
         num_mat += 2

      if 'precond' in key_list:
         matrix_set['precond'] = all_matrices[num_mat]
         matrix_set['precond_p'] = all_matrices[num_mat + 1]

   return matrix_set
