from abc    import ABC, abstractmethod

from numpy import ndarray

class RHS(ABC):
   def __init__(self, shape: tuple[int, ...],  *params, **kwparams) -> None:
      self.shape = shape
      self.params = params
      self.kwparams = kwparams

   def __call__(self, vec: ndarray) -> ndarray:
      """Compute the value of the right-hand side based on the input state.

      :param vec: Vector containing the input state. It can have any shape, as long as its size is the same as the
                  one used to create this RHS object
      :return: Value of the right-hand side, in the same shape as the input
      """
      old_shape = vec.shape
      result = self.__compute_rhs__(vec.reshape(self.shape), *self.params, **self.kwparams)
      return result.reshape(old_shape)

   @abstractmethod
   def __compute_rhs__(self, vec: ndarray, *params, **kwparams) -> ndarray:
      pass
