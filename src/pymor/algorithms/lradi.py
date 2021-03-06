# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.lyapunov import _solve_lyap_lrcf_check_args
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.tools.random import get_random_state
from pymor.vectorarrays.constructions import cat_arrays


@defaults('lradi_tol', 'lradi_maxiter', 'lradi_shifts', 'projection_shifts_init_maxiter',
          'projection_shifts_init_seed', 'projection_shifts_subspace_columns')
def lyap_lrcf_solver_options(lradi_tol=1e-10,
                             lradi_maxiter=500,
                             lradi_shifts='projection_shifts',
                             projection_shifts_init_maxiter=20,
                             projection_shifts_init_seed=None,
                             projection_shifts_subspace_columns=6):
    """Return available Lyapunov solvers with default options.

    Parameters
    ----------
    lradi_tol
        See :func:`solve_lyap_lrcf`.
    lradi_maxiter
        See :func:`solve_lyap_lrcf`.
    lradi_shifts
        See :func:`solve_lyap_lrcf`.
    projection_shifts_init_maxiter
        See :func:`projection_shifts_init`.
    projection_shifts_init_seed
        See :func:`projection_shifts_init`.
    projection_shifts_subspace_columns
        See :func:`projection_shifts`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'lradi': {'type': 'lradi',
                      'tol': lradi_tol,
                      'maxiter': lradi_maxiter,
                      'shifts': lradi_shifts,
                      'shift_options':
                      {'projection_shifts': {'type': 'projection_shifts',
                                             'init_maxiter': projection_shifts_init_maxiter,
                                             'init_seed': projection_shifts_init_seed,
                                             'subspace_columns': projection_shifts_subspace_columns}}}}


def solve_lyap_lrcf(A, E, B, trans=False, options=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    See :func:`pymor.algorithms.lyapunov.solve_lyap_lrcf` for a
    general description.

    This function uses the low-rank ADI iteration as described in
    Algorithm 4.3 in :cite:`PK16`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the Lyapunov equation is
        transposed.
    options
        The solver options to use (see
        :func:`lyap_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'lradi', None, False)
    logger = getLogger('pymor.algorithms.lradi.solve_lyap_lrcf')

    shift_options = options['shift_options'][options['shifts']]
    if shift_options['type'] == 'projection_shifts':
        init_shifts = projection_shifts_init
        iteration_shifts = projection_shifts
    else:
        raise ValueError('Unknown low-rank ADI shift strategy.')

    if E is None:
        E = IdentityOperator(A.source)

    Z = A.source.empty()
    W = B.copy()

    j = 0
    j_shift = 0
    shifts = init_shifts(A, E, W, shift_options)
    res = np.linalg.norm(W.gramian(), ord=2)
    init_res = res
    Btol = res * options['tol']

    while res > Btol and j < options['maxiter']:
        if shifts[j_shift].imag == 0:
            AaE = A + shifts[j_shift].real * E
            if not trans:
                V = AaE.apply_inverse(W)
                W -= E.apply(V) * (2 * shifts[j_shift].real)
            else:
                V = AaE.apply_inverse_adjoint(W)
                W -= E.apply_adjoint(V) * (2 * shifts[j_shift].real)
            Z.append(V * np.sqrt(-2 * shifts[j_shift].real))
            j += 1
        else:
            AaE = A + shifts[j_shift] * E
            gs = -4 * shifts[j_shift].real
            d = shifts[j_shift].real / shifts[j_shift].imag
            if not trans:
                V = AaE.apply_inverse(W)
                W += E.apply(V.real + V.imag * d) * gs
            else:
                V = AaE.apply_inverse_adjoint(W).conj()
                W += E.apply_adjoint(V.real + V.imag * d) * gs
            g = np.sqrt(gs)
            Z.append((V.real + V.imag * d) * g)
            Z.append(V.imag * (g * np.sqrt(d**2 + 1)))
            j += 2
        j_shift += 1
        res = np.linalg.norm(W.gramian(), ord=2)
        logger.info(f'Relative residual at step {j}: {res/init_res:.5e}')
        if j_shift >= shifts.size:
            shifts = iteration_shifts(A, E, V, Z, shifts, shift_options)
            j_shift = 0

    if res > Btol:
        logger.warning(f'Prescribed relative residual tolerance was not achieved '
                       f'({res/init_res:e} > {options["tol"]:e}) after ' f'{options["maxiter"]} ADI steps.')

    return Z


def projection_shifts_init(A, E, B, shift_options):
    """Find starting projection shifts.

    Uses Galerkin projection on the space spanned by the right-hand side if
    it produces stable shifts.
    Otherwise, uses a randomly generated subspace.
    See :cite:`PK16`, pp. 92-95.

    Parameters
    ----------
    A
        The |Operator| A from the corresponding Lyapunov equation.
    E
        The |Operator| E from the corresponding Lyapunov equation.
    B
        The |VectorArray| B from the corresponding Lyapunov equation.
    shift_options
        The shift options to use (see :func:`lyap_lrcf_solver_options`).

    Returns
    -------
    shifts
        A |NumPy array| containing a set of stable shift parameters.
    """
    random_state = get_random_state(seed=shift_options['init_seed'])
    for i in range(shift_options['init_maxiter']):
        Q = gram_schmidt(B, atol=0, rtol=0)
        shifts = spla.eigvals(A.apply2(Q, Q), E.apply2(Q, Q))
        shifts = shifts[shifts.real < 0]
        if shifts.size == 0:
            # use random subspace instead of span{B} (with same dimensions)
            B = B.random(len(B), distribution='normal', random_state=random_state)
        else:
            return shifts
    raise RuntimeError('Could not generate initial shifts for low-rank ADI iteration.')


def projection_shifts(A, E, V, Z, prev_shifts, shift_options):
    """Find further projection shifts.

    Uses Galerkin projection on spaces spanned by LR-ADI iterates.
    See :cite:`PK16`, pp. 92-95.

    Parameters
    ----------
    A
        The |Operator| A from the corresponding Lyapunov equation.
    E
        The |Operator| E from the corresponding Lyapunov equation.
    V
        A |VectorArray| representing the currently computed iterate.
    Z
        A |VectorArray| representing the current approximate solution.
    prev_shifts
        A |NumPy array| containing the set of all previously used shift
        parameters.
    shift_options
        The shift options to use (see :func:`lyap_lrcf_solver_options`).

    Returns
    -------
    shifts
        A |NumPy array| containing a set of stable shift parameters.
    """
    if shift_options['subspace_columns'] == 1:
        if prev_shifts[-1].imag != 0:
            Q = gram_schmidt(cat_arrays([V.real, V.imag]), atol=0, rtol=0)
        else:
            Q = gram_schmidt(V, atol=0, rtol=0)
    else:
        num_columns = shift_options['subspace_columns'] * len(V)
        Q = gram_schmidt(Z[-num_columns:], atol=0, rtol=0)

    shifts = spla.eigvals(A.apply2(Q, Q), E.apply2(Q, Q))
    shifts = shifts[shifts.real < 0]
    shifts = shifts[shifts.imag >= 0]
    if shifts.size == 0:
        return prev_shifts
    else:
        shifts.imag[-shifts.imag / shifts.real < 1e-12] = 0
        shifts = shifts[np.abs(shifts).argsort()]
        return shifts
