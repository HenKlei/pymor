# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


def thermal_block_example(diameter=1/100):
    """Return 2x2 thermal block example.

    Parameters
    ----------
    diameter
        Grid element diameter.

    Returns
    -------
    fom
        Thermal block problem as a |StationaryModel|.
    """
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.builtin import discretize_stationary_cg

    p = thermal_block_problem((2, 2))
    fom, _ = discretize_stationary_cg(p, diameter=diameter)
    return fom

def penzl_example():
    """Return Penzl's example.

    Returns
    -------
    fom
        Penzl's FOM example as an |LTIModel|.
    """
    import numpy as np
    import scipy.sparse as sps

    from pymor.models.iosys import LTIModel

    n = 1006
    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.ones((n, 1))
    B[:6] = 10
    C = B.T
    fom = LTIModel.from_matrices(A, B, C)

    return fom

def penzl_mimo_example(n, m=2, p=3):
    """Return modified multiple-input multiple-output Penzl's example.

    Parameters
    ----------
    n
        Model order.

    Returns
    -------
    fom
        Penzl's FOM example as an |LTIModel|.
    """
    import numpy as np
    import scipy.sparse as sps

    from pymor.models.iosys import LTIModel

    A1 = np.array([[-1, 100], [-100, -1]])
    A2 = np.array([[-1, 200], [-200, -1]])
    A3 = np.array([[-1, 400], [-400, -1]])
    A4 = sps.diags(np.arange(-1, -n + 5, -1))
    A = sps.block_diag((A1, A2, A3, A4))
    B = np.arange(m*n).reshape(n, m)
    C = np.arange(p*n).reshape(p, n)
    return LTIModel.from_matrices(A, B, C)

def msd_example(n=6, m=2, m_i=4, k_i=4, c_i=1, as_lti=False):
    """Mass-spring-damper model as (port-Hamiltonian) linear time-invariant system.

    Taken from :cite:`GPBV12`.

    Parameters
    ----------
    n
        The order of the model.
    m
        The number or inputs and outputs of the model.
    m_i
        The weight of the masses.
    k_i
        The stiffness of the springs.
    c_i
        The amount of damping.
    as_lti
        If `True`, the matrices of the standard linear time-invariant system are returned.
        Otherwise, the matrices of the port-Hamiltonian linear time-invariant system are returned.

    Returns
    -------
    fom
        Mass-spring-damper model as an |LTIModel| (if `as_lti` is `True`)
        or |PHLTIModel| (if `as_lti` is `False`).
    """
    import numpy as np
    import scipy.linalg as spla

    from pymor.models.iosys import LTIModel, PHLTIModel

    assert n % 2 == 0
    n //= 2

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    if m == 2:
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0], [0, 0, 0, 1 / m_i, 0, 0]])
    elif m == 1:
        B = np.array([[0, 1, 0, 0, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])
    else:
        assert False

    J_i = np.array([[0, 1], [-1, 0]])
    J = np.kron(np.eye(3), J_i)
    R_i = np.array([[0, 0], [0, c_i]])
    R = np.kron(np.eye(3), R_i)

    for i in range(4, n + 1):
        B = np.vstack((B, np.zeros((2, m))))
        C = np.hstack((C, np.zeros((m, 2))))

        J = np.block([
            [J, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), J_i]
        ])

        R = np.block([
            [R, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), R_i]
        ])

        A = np.block([
            [A, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, i * 2))]
        ])

        A[2 * i - 2, 2 * i - 2] = 0
        A[2 * i - 1, 2 * i - 1] = -c_i / m_i
        A[2 * i - 3, 2 * i - 2] = k_i
        A[2 * i - 2, 2 * i - 1] = 1 / m_i
        A[2 * i - 2, 2 * i - 3] = 0
        A[2 * i - 1, 2 * i - 2] = -2 * k_i
        A[2 * i - 1, 2 * i - 4] = k_i

    Q = spla.solve(J - R, A)
    G = B
    D = np.zeros((m, m))
    S = (D + D.T) / 2
    N = -(D - D.T) / 2

    if as_lti:
        return LTIModel.from_matrices(A, B, C, D)

    return PHLTIModel.from_matrices(J, R, G, S=S, N=N, Q=Q)

def transfer_function_delay_example(tau=1, a=-0.1):
    """Return transfer function of a 1D system with input delay.

    Parameters
    ----------
    tau
        Time delay.
    a
        The matrix A in the 1D system as a scalar.

    Returns
    -------
    tf
        Delay model as a |TransferFunction|.
    """
    import numpy as np

    from pymor.models.transfer_function import TransferFunction

    def H(s):
        return np.array([[np.exp(-tau * s) / (s - a)]])

    def dH(s):
        return np.array([[(-tau*s + tau*a - 1) * np.exp(-tau * s) / (s - a) ** 2]])

    tf = TransferFunction(1, 1, H, dH)

    return tf

def heat_equation_example(grid_intervals=50, nt=50):
    """Return heat equation example with a high-conductivity and two parametrized channels.

    Parameters
    ----------
    grid_intervals
        Number of intervals in each direction of the two-dimensional |RectDomain|.
    nt
        Number of time steps.

    Returns
    -------
    fom
        Heat equation problem as an |InstationaryModel|.
    """
    from pymor.analyticalproblems.domaindescriptions import RectDomain
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.discretizers.builtin import discretize_instationary_cg
    from pymor.parameters.functionals import ExpressionParameterFunctional

    # setup analytical problem
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(top='dirichlet', bottom='neumann'),

            diffusion=LincombFunction(
                [ConstantFunction(1., dim_domain=2),
                 ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                    dim_domain=2),
                 ExpressionFunction('(0.35 < x[0] < 0.40) * (x[1] > 0.3) * 1. + '
                                    '(0.60 < x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                    dim_domain=2)],
                [1.,
                 100. - 1.,
                 ExpressionParameterFunctional('top[0] - 1.', {'top': 1})]
            ),

            rhs=ConstantFunction(value=100., dim_domain=2) * ExpressionParameterFunctional('sin(10*pi*t[0])', {'t': 1}),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),

            neumann_data=ExpressionFunction('(0.45 < x[0] < 0.55) * -1000.', dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction('(0.45 < x[0] < 0.55) * (x[1] < 0.7) * 10.', dim_domain=2)
    )

    # discretize using continuous finite elements
    fom, _ = discretize_instationary_cg(analytical_problem=problem, diameter=1./grid_intervals, nt=nt)

    return fom


def heat_equation_non_parametric_example(diameter=0.1, nt=100):
    """Return non-parametric heat equation example with one output.

    Parameters
    ----------
    diameter
        Diameter option for the domain discretizer.
    nt
        Number of time steps.

    Returns
    -------
    fom
        Heat equation problem as an |InstationaryModel|.
    """
    from pymor.analyticalproblems.domaindescriptions import RectDomain
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.discretizers.builtin import discretize_instationary_cg

    p = InstationaryProblem(
        StationaryProblem(
            domain=RectDomain([[0., 0.], [1., 1.]], left='robin', right='robin', top='robin', bottom='robin'),
            diffusion=ConstantFunction(1., 2),
            robin_data=(ConstantFunction(1., 2), ExpressionFunction('(x[0] < 1e-10) * 1.', 2)),
            outputs=[('l2_boundary', ExpressionFunction('(x[0] > (1 - 1e-10)) * 1.', 2))]
        ),
        ConstantFunction(0., 2),
        T=1.
    )

    fom, _ = discretize_instationary_cg(p, diameter=diameter, nt=nt)

    return fom


def heat_equation_1d_example(diameter=0.01, nt=100):
    """Return parametric 1D heat equation example with one output.

    Parameters
    ----------
    diameter
        Diameter option for the domain discretizer.
    nt
        Number of time steps.

    Returns
    -------
    fom
        Heat equation problem as an |InstationaryModel|.
    """
    from pymor.analyticalproblems.domaindescriptions import LineDomain
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction, LincombFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.discretizers.builtin import discretize_instationary_cg
    from pymor.parameters.functionals import ProjectionParameterFunctional

    p = InstationaryProblem(
        StationaryProblem(
            domain=LineDomain([0., 1.], left='robin', right='robin'),
            diffusion=LincombFunction([ExpressionFunction('(x[0] <= 0.5) * 1.', 1),
                                       ExpressionFunction('(0.5 < x[0]) * 1.', 1)],
                                      [1,
                                       ProjectionParameterFunctional('diffusion')]),
            robin_data=(ConstantFunction(1., 1), ExpressionFunction('(x[0] < 1e-10) * 1.', 1)),
            outputs=(('l2_boundary', ExpressionFunction('(x[0] > (1 - 1e-10)) * 1.', 1)),),
        ),
        ConstantFunction(0., 1),
        T=3.
    )

    fom, _ = discretize_instationary_cg(p, diameter=diameter, nt=nt)

    return fom
