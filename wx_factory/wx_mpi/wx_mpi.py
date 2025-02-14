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
    In such a case, only the processes members of that communicator must enter the context.
    *There is a synchronization point when exiting this context.*

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
        self.should_skip = self.rank != self.root_rank

    def __enter__(self):
        "Do nothing (but must be implemented)."
        return self

    def __exit__(self, exception_type, *_):
        """Broadcast result from the single process that actually worked inside the context. If that
        particular process raised an exception, all participating processes will also raise one.
        The non-working processes will simply exit, without printing a stack trace

        :return: Whether to ignore the potential exception that was thrown from inside the context
        :rtype: bool
        """
        is_ok = exception_type is None or self.should_skip  # No error, or not a rank that was doing stuff
        transmit = self.return_value if is_ok else _SkippableFailure()
        self.return_value = self.comm.bcast(transmit, root=self.root_rank)

        ignore_error = self.return_value is None or not isinstance(self.return_value, _SkippableFailure)
        return ignore_error


class MultipleProcesses:
    def __init__(self, comm: MPI.Comm = MPI.COMM_WORLD, num_procs: int = 0):
        self.comm = comm

        if num_procs > 0:
            self.should_skip = self.comm.rank >= num_procs
            self.sub_comm = comm.Split(0 if self.should_skip else 1, self.comm.rank)
        else:
            self.should_skip = False
            self.sub_comm = self.comm

    def __enter__(self):
        "Do nothing (but must be implemented)."
        return self

    def __exit__(self, exception_type, *_):
        num_errors = 0
        if exception_type is not None and not issubclass(exception_type, _Skip):
            num_errors = 1

        total_errors = self.comm.allreduce(num_errors)

        return total_errors == 0


class Conditional:
    """When used in combination with :class:`SingleProcess`, skips the content of the context for everyone
    except the root, as defined by the associated :class:`SingleProcess`
    """

    def __init__(self, context: SingleProcess):
        """Create a context manager to temporarily disable a set of processes

        :param context: Driver for this context. It determines whether we should skip execution (i.e. raise
            a _Skip exception)
        :type context: SingleProcess
        """
        self.should_skip = context.should_skip

    def __enter__(self):
        """If this process is not supposed to work in the context, raise a :class:`_Skip` exception,
        which will be caught by the associated :class:`SingleProcess`. Otherwise just proceed normally.

        :raises _Skip: To be caught by :class:`SingleProcess`, to disable this process during the context
        :return: self
        """
        if self.should_skip:
            raise _Skip

        return self

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
