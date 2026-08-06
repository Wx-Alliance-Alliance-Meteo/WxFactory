class SolverInfo:
    def __init__(
        self,
        flag: int = 0,
        time: float = 0.0,
        total_num_it: int = 0,
        iterations: list[tuple[float, float, float]] | None = None,
    ) -> None:
        self.flag = flag
        self.time = time
        self.total_num_it = total_num_it
        self.iterations = iterations if iterations is not None else []
