"""Different methods to solve a system of equations."""
from .fgmres            import fgmres
from .gcrot             import gcrot
from .kiops             import kiops
from .global_operations import global_dotprod, global_inf_norm, global_norm
from .matvec            import MatvecOp, MatvecOpBasic, MatvecOpRat, matvec_fun, matvec_rat
from .nonlin            import KrylovJacobian, newton_krylov
from .pmex              import pmex
from .solver_info       import SolverInfo

__all__ = ['fgmres', 'kiops', 'global_dotprod', 'global_inf_norm', 'global_norm', 'KrylovJacobian',
           'MatvecOp', 'MatvecOpBasic', 'MatvecOpRat',
           'matvec_fun', 'matvec_rat', 'newton_krylov', 'pmex', 'SolverInfo']
