class MissingModule:
    """Stand-in for an optional dependency that failed to import.

    Any attribute access on the instance raises the original import error, so
    the failure is reported at the point of use rather than at import time."""

    def __init__(self, error) -> None:
        self.error = error

    def __getattr__(self, name):
        raise self.error
