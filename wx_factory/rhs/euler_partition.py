import torch


class EulerRhsPartition:
    def __init__(self, rhs, partition):
        if partition not in ("explicit", "implicit"):
            raise ValueError(f"Unknown Euler partition '{partition}'")
        self.rhs = rhs
        self.partition = partition
        self.metric = rhs.metric
        self.evaluate = rhs.explicit if partition == "explicit" else rhs.implicit

    def __call__(self, state):
        return self.evaluate(state)

    def linearize(self, state):
        if self.partition == "explicit":
            return self.rhs.linearize(state).explicit
        from ..jacobian.euler_vertical import EulerImplicitLinearization

        return EulerImplicitLinearization(self.rhs, state)

    def double_precision_rhs(self):
        return EulerRhsPartition(self.rhs.double_precision_rhs(), self.partition)

    def jacobian_rhs(self, state):
        return self.double_precision_rhs()(state.to(torch.float64))
