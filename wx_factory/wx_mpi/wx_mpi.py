"""A set of functions and classes to simplify MPI usage."""

from typing import Any

from mpi4py import MPI


class _Skip(SystemExit):
    """Exception that indicates this process is inactive in the associated SingleProcess context."""


class _SkippableFailure(Exception):
    """Exception that indicates the active process of a SingleProcess context has failed."""


class SingleProcess:
    """
    Class to use in combination with :class:`Conditional` to create a context with a set of processes where
    only one of them will execute the body. Optionally, that process can return something that
    will be broadcast to every other process in the context. By default, all processes in MPI.COMM_WORLD
    must participate in the context, but it is possible to pass a communicator when creating the context.

    .. code-block:: python
        :caption: Basic usage

        with SingleProcess() as s, Conditional(s):
            print(f"Only rank {s.rank} is executing this")


    .. code-block:: python
        :caption: With a return value

        with SingleProcess() as s, Conditional(s):
            s.return_value = "Hi there!"

        # Now everyone has access to s.return_value
        print(f"{s.return_value} from process {MPI.COMM_WORLD.rank}")

    .. code-block:: python
        :caption: With a custom communicator

        with SingleProcess(comm=my_comm) as s, Conditional(s):  # Every process in `my_comm` must enter here
            do_stuff()


    :param comm: Communicator that groups all processes (and only these) participating in the context
    :type comm: MPI.Comm

    :param root_rank: Rank of the process that will perform the work in the context
    :type root_rank: int

    :param rank: Rank of this process within the communicator
    :type rank: int

    :param return_value: Value that will be transmitted to all participating processes when
        exiting the context
    :type return_value: Any

    """

    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD, root_rank: int = 0):
        """Create the context manager

        :param comm: Communicator that groups participating processes, defaults to MPI.COMM_WORLD
        :type comm: MPI.Comm, optional
        :param root_rank: Rank of the one process that will do the work, defaults to 0
        :type root_rank: int, optional
        """
        self.comm = comm
        self.rank = self.comm.rank
        self.root_rank = root_rank
        self.return_value = None

    def __enter__(self):
        """Nothing special done here

        :return: self
        """
        return self

    def __exit__(self, exception_type, *_):
        """Broadcast result from the single process that actually worked inside the context. If that
        particular process raised an exception, all participating processes will also raise one.
        The non-working processes will simply exit, without printing a stack trace

        :return: Whether to ignore the potential exception that was thrown from inside the context
        :rtype: bool
        """
        is_ok = exception_type is None or self.root_rank != self.rank  # No error, or not a rank that was doing stuff
        transmit = self.return_value if is_ok else _SkippableFailure()
        self.return_value = self.comm.bcast(transmit, root=self.root_rank)

        ignore_error = self.return_value is None or not isinstance(self.return_value, _SkippableFailure)
        return ignore_error


class Conditional:
    """When used in combination with :class:`SingleProcess`, skips the content of the context for everyone
    except the root, as defined by the associated :class:`SingleProcess`
    """

    def __init__(self, context: SingleProcess):
        """Create a context manager to temporarily disable a set of processes

        :param context: Driver for this context
        :type context: SingleProcess
        """
        self.comm = context.comm
        self.root_rank = context.root_rank
        self.rank = self.comm.rank

    def __enter__(self):
        """If this process is not supposed to work in the context, raise a :class:`_Skip` exception,
        which will be caught by the associated :class:`SingleProcess`. Otherwise just proceed normally.

        :raises _Skip: To be caught by :class:`SingleProcess`, to disable this process during the context
        :return: self
        """
        if self.rank == self.root_rank:
            return self
        else:
            raise _Skip

    def __exit__(self, *_):
        pass


def do_once(action, *args, comm: MPI.Comm = MPI.COMM_WORLD) -> Any:
    """Perform the given action with exactly one process in the communicator

    :param action: Function to be called by the active process
    :type action: Callable
    :param comm: Communicator that groups exactly every process that call this function, defaults to MPI.COMM_WORLD
    :type comm: MPI.Comm, optional
    :return: The result of `action`, from every process in `comm`
    :rtype: Any
    """
    with SingleProcess(comm) as s, Conditional(s):
        s.return_value = action(*args)

    return s.return_value
