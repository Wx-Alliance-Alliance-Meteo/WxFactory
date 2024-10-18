"""Different methods to solve a system of equations."""
from .fgmres            import fgmres
from .gcrot             import gcrot
from .kiops             import kiops
from .kiops_nest        import kiops_nest
from .global_operations import global_dotprod, global_inf_norm, global_norm
from .matvec            import MatvecOp, MatvecOpBasic, MatvecOpRat, matvec_fun, matvec_rat
from .nonlin            import KrylovJacobian, newton_krylov
from .pmex              import pmex
from .pmex_1s           import pmex_1s
from .pmex_ne1s         import pmex_ne1s
from .cwy_1s            import cwy_1s
from .cwy_ne            import cwy_ne
from .cwy_ne1s          import cwy_ne1s
from .icwy_1s           import icwy_1s
from .icwy_ne           import icwy_ne
from .icwy_ne1s         import icwy_ne1s
from .icwy_neiop        import icwy_neiop
from .dcgs2             import dcgs2
from .solver_info       import SolverInfo

__all__ = ['fgmres', 'kiops', 'global_dotprod', 'global_inf_norm', 'global_norm', 'KrylovJacobian',
           'MatvecOp', 'MatvecOpBasic', 'MatvecOpRat',
           'matvec_fun', 'matvec_rat', 'newton_krylov', 'pmex', 'SolverInfo',
           'icwy_ne', 'icwy_1s', 'icwy_ne1s', 'icwy_neiop', 'kiops_nest', 
           'cwy_1s', 'cwy_ne', 'cwy_ne1s', 'dcgs2', 'pmex_1s', 'pmex_ne1s']
