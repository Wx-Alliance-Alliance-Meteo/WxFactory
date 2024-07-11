from abc    import ABC, abstractmethod

from numpy.typing import NDArray

class RHS(ABC):
   def __init__(self, shape: tuple[int, ...],  *params, **kwparams) -> None:
      self.shape = shape
      self.params = params
      self.kwparams = kwparams

   def __call__(self, vec: NDArray) -> NDArray:
      old_shape = vec.shape
      result = self.__compute_rhs__(vec.reshape(self.shape), *self.params, **self.kwparams)
      return result.reshape(old_shape)

   @abstractmethod
   def __compute_rhs__(self, vec: NDArray, *params, **kwparams) -> NDArray:
      pass
