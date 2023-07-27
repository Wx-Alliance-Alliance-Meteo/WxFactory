import numpy
from .explicit_runge_kutta import RungeKutta

class ERK3_2_3L(RungeKutta):
    # effective number of stages
    n_stages = 3

    # order of the main method
    order = 3

    # order of the secondary embedded method
    error_estimator_order = 2 #for both linear and non-linear

    # time fraction coefficients (nodes)
    C = numpy.array([0., 1./2., 1.])

    # runge kutta coefficient matrix
    A = numpy.array([[0., 0., 0.], \
      [1/2, 0., 0.], \
      [-1., 2., 0.]])

    # output coefficients (weights)
    B = numpy.array([1./6., 2./3., 1./6.])

    # error coefficients (weights Bh - B)
    E = numpy.array([1./12., -1./6., 1./12.,0])

    # dense output
    # P = numpy.array([[4655552711362/22874653954995, -18682724506714/9892148508045, 34259539580243/13192909600954, 584795268549/6622622206610],
    #  [-215264564351/13552729205753, 17870216137069/13817060693119, -28141676662227/17317692491321, 2508943948391/7218656332882]])
    
    # Parameters for stepsize control
    # sc_params = "W"



class ARK3_2_4L_2_SA_ERK(RungeKutta):

    # effective number of stages
    n_stages = 4

    # order of the main method
    order = 3

    # order of the secondary embedded method
    error_estimator_order = 2

    # time fraction coefficients (nodes)
    C = numpy.array([0., 1767732205903/2027836641118, 3/5, 1])

    # runge kutta coefficient matrix
    A = numpy.array([[0., 0., 0., 0.], \
      [1767732205903/2027836641118, 0., 0., 0.], \
      [5535828885825/10492691773637, 788022342437/10882634858940, 0., 0.], \
      [6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841, 0.]])

    # output coefficients (weights)
    B = numpy.array([1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, 1767732205903/4055673282236])

    # error coefficients (weights Bh - B)
    E = numpy.array([ 0.02709926,  0.11013521, -0.10306493, -0.03416955, 0])

class Merson4(RungeKutta):

    # effective number of stages
    n_stages = 5

    # order of the main method
    order = 4

    # order of the secondary embedded method
    error_estimator_order = 3

    # time fraction coefficients (nodes)
    C = numpy.array([0, 1/3, 1/3, 1/2, 1])

    # runge kutta coefficient matrix
    A = numpy.array([
        [0, 0, 0, 0, 0],
        [1/3, 0, 0, 0, 0],
        [1/6, 1/6, 0, 0, 0],
        [1/8, 0, 3/8, 0, 0],
        [1/2, 0, -3/2, 2, 0]])

    # output coefficients (weights)
    B = numpy.array([1/6, 0, 0, 2/3, 1/6])

    # error coefficients (weights Bh - B)
    E = numpy.array([1/10, 0, 3/10, 2/5, 1/5, 0])        # B_hat
    E[:-1] -= B

class RK23(RungeKutta):

    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = numpy.array([0, 1/2, 3/4])
    A = numpy.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 3/4, 0]
    ])
    B = numpy.array([2/9, 1/3, 4/9])
    E = numpy.array([5/72, -1/12, -1/9, 1/8])


class RK45(RungeKutta):
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = numpy.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = numpy.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = numpy.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = numpy.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])


METHODS = {'RK23'              : RK23,
           'RK45'              : RK45,
           'MERSON4'           : Merson4,
           'ARK3(2)4L[2]SA-ERK': ARK3_2_4L_2_SA_ERK,
           'ERK3(2)3L'         : ERK3_2_3L}
