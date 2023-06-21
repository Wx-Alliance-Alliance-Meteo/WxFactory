from abc import ABC, abstractmethod
import numpy as np

# typing
from typing import Self, Any, Literal, TypeVar, Sequence, overload
from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)

Contiguity = Literal['C', 'F']
InheritContiguity = Literal['C', 'F', 'A', 'K']


class ArrayModule(ABC):

    @abstractmethod
    def empty(self: Self,
              shape: int | Sequence[int],
              dtype: type[T] = np.float64,
              order: Contiguity = 'C',
              *,
              like: NDArray | None = None) -> NDArray[T]: ...

    @overload
    def empty_like(self: Self,
                   prototype: NDArray[T],
                   dtype: None = None,
                   order: InheritContiguity = 'K',
                   subok: bool = True,
                   shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @overload
    def empty_like(self: Self,
                   prototype: NDArray,
                   dtype: type[T],
                   order: InheritContiguity = 'K',
                   subok: bool = True,
                   shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @abstractmethod
    def empty_like(self: Self, *args): ...

    @abstractmethod
    def eye(self: Self,
            N: int,
            M: int | None = None,
            k: int = 0,
            dtype: type[T] = np.float64,
            order: Contiguity = 'C',
            *,
            like: NDArray | None = None) -> NDArray[T]: ...
    
    @abstractmethod
    def identity(self: Self,
                 n: int,
                 dtype: type[T] = np.float64,
                 *,
                 like: NDArray | None = None) -> NDArray[T]: ...
    
    @abstractmethod
    def ones(self: Self,
             shape: int | Sequence[int],
             dtype: type[T] = np.float64,
             order: Contiguity = 'C',
             *,
             like: NDArray | None = None) -> NDArray[T]: ...
    
    @overload
    def ones_like(self: Self,
                  a: NDArray[T],
                  dtype: None = None,
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @overload
    def ones_like(self: Self,
                  a: NDArray,
                  dtype: type[T],
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @abstractmethod
    def ones_like(self: Self, *args): ...

    @abstractmethod
    def zeros(self: Self,
             shape: int | Sequence[int],
             dtype: type[T] = np.float64,
             order: Contiguity = 'C',
             *,
             like: NDArray | None = None) -> NDArray[T]: ...
    
    @overload
    def zeros_like(self: Self,
                  a: NDArray[T],
                  dtype: None = None,
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @overload
    def zeros_like(self: Self,
                  a: NDArray,
                  dtype: type[T],
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @abstractmethod
    def zeros_like(self: Self, *args): ...

    @overload
    def full(self: Self,
             shape: int | Sequence[int],
             fill_value: T | NDArray[T],
             dtype: None = None,
             order: Contiguity = 'C',
             *,
             like: NDArray | None = None) -> NDArray[T]: ...
    @overload
    def full(self: Self,
             shape: int | Sequence[int],
             fill_value: Any,
             dtype: type[T],
             order: Contiguity = 'C',
             *,
             like: NDArray | None = None) -> NDArray[T]: ...
    @abstractmethod
    def full(self: Self, *args): ...

    @overload
    def full_like(self: Self,
                  a: NDArray[T],
                  fill_value: Any,
                  dtype: None = None,
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @overload
    def full_like(self: Self,
                  a: NDArray,
                  fill_value: Any,
                  dtype: type[T],
                  order: InheritContiguity = 'K',
                  subok: bool = True,
                  shape: int | Sequence[int] | None = None) -> NDArray[T]: ...
    @abstractmethod
    def full_like(self: Self, *args): ...

    # TODO: asarray should accept nested sequences
    @overload
    def asarray(self: Self,
                a: NDArray[T] | Sequence[T],
                dtype: None = None,
                order: InheritContiguity = 'K',
                *,
                like: NDArray | None = None) -> NDArray[T]: ...
    @overload
    def asarray(self: Self,
                a: NDArray | Sequence,
                dtype: type[T],
                order: InheritContiguity = 'K',
                *,
                like: NDArray | None = None) -> NDArray[T]: ...
    @abstractmethod
    def asarray(self: Self, *args): ...

    @abstractmethod
    def linspace(self: Self,
                 start: int,
                 stop: int,
                 num: int = 50,
                 endpoint: bool = True,
                 retstep: Literal[False] = False,
                 dtype: type[T] = np.float64,
                 axis: int = 0) -> NDArray[T]: ...
    