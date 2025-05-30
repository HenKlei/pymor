# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.algorithms.lincomb import assemble_lincomb
from pymor.operators.constructions import LowRankOperator, LowRankUpdatedOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


def construct_operators_and_vectorarrays(m, n, r, k, rng):
    space_m = NumpyVectorSpace(m)
    space_n = NumpyVectorSpace(n)
    A = NumpyMatrixOperator(rng.normal(size=(m, n)))
    L = space_m.random(r, distribution='normal')
    C = rng.normal(size=(r, r))
    R = space_n.random(r, distribution='normal')
    U = space_n.random(k, distribution='normal')
    V = space_m.random(k, distribution='normal')
    return A, L, C, R, U, V


def test_low_rank_apply(rng):
    _, L, C, R, U, _ = construct_operators_and_vectorarrays(6, 5, 2, 3, rng)

    LR = LowRankOperator(L, C, R)
    V = LR.apply(U)
    assert np.allclose(V.to_numpy(), L.to_numpy() @ C @ (R.to_numpy().T @ U.to_numpy()))

    LR = LowRankOperator(L, C, R, inverted=True)
    V = LR.apply(U)
    assert np.allclose(V.to_numpy(),
                       L.to_numpy() @ spla.solve(C, R.to_numpy().T @ U.to_numpy()))


def test_low_rank_apply_adjoint(rng):
    _, L, C, R, _, V = construct_operators_and_vectorarrays(6, 5, 2, 3, rng)

    LR = LowRankOperator(L, C, R)
    U = LR.apply_adjoint(V)
    assert np.allclose(U.to_numpy(), R.to_numpy() @ C.T @ (L.to_numpy().T @ V.to_numpy()))

    LR = LowRankOperator(L, C, R, inverted=True)
    U = LR.apply_adjoint(V)
    assert np.allclose(U.to_numpy(),
                       R.to_numpy() @ spla.solve(C.T, L.to_numpy().T @ V.to_numpy()))


def test_low_rank_updated_apply_inverse(rng):
    A, L, C, R, _, V = construct_operators_and_vectorarrays(5, 5, 2, 3, rng)

    LR = LowRankOperator(L, C, R)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    U = op.apply_inverse(V)
    mat = A.matrix + L.to_numpy() @ C @ R.to_numpy().T
    assert np.allclose(U.to_numpy(), spla.solve(mat, V.to_numpy()))

    LR = LowRankOperator(L, C, R, inverted=True)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    U = op.apply_inverse(V)
    mat = A.matrix + L.to_numpy() @ spla.solve(C, R.to_numpy().T)
    assert np.allclose(U.to_numpy(), spla.solve(mat, V.to_numpy()))


def test_low_rank_updated_apply_inverse_adjoint(rng):
    A, L, C, R, U, _ = construct_operators_and_vectorarrays(5, 5, 2, 3, rng)

    LR = LowRankOperator(L, C, R)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    V = op.apply_inverse_adjoint(U)
    mat = A.matrix + L.to_numpy() @ C @ R.to_numpy().T
    assert np.allclose(V.to_numpy(), spla.solve(mat.T, U.to_numpy()))

    LR = LowRankOperator(L, C, R, inverted=True)
    op = LowRankUpdatedOperator(A, LR, 1, 1)
    V = op.apply_inverse_adjoint(U)
    mat = A.matrix + L.to_numpy() @ spla.solve(C, R.to_numpy().T)
    assert np.allclose(V.to_numpy(), spla.solve(mat.T, U.to_numpy()))


def test_low_rank_assemble(rng):
    r1, r2 = 2, 3
    _, L1, C1, R1, _, _ = construct_operators_and_vectorarrays(5, 5, r1, 0, rng)
    _, L2, C2, R2, _, _ = construct_operators_and_vectorarrays(5, 5, r2, 0, rng)

    LR1 = LowRankOperator(L1, C1, R1)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert not op.inverted

    op = (LR1 + (LR1 + LR2) + LR2).assemble()
    assert isinstance(op, LowRankOperator)

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2, inverted=True)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert op.inverted

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert op is None


def test_low_rank_updated_assemble(rng):
    A, L, C, R, _, _ = construct_operators_and_vectorarrays(5, 5, 2, 0, rng)
    LR = LowRankOperator(L, C, R)

    op = (A + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + LR + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + (A + LR).assemble() + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)


def test_low_rank_updated_assemble_apply(rng):
    A, L, C, R, U, _ = construct_operators_and_vectorarrays(5, 5, 2, 3, rng)

    LR = LowRankOperator(L, C, R)
    op = (A + (A + LR).assemble() + LR).assemble()
    V = op.apply(U)
    assert np.allclose(V.to_numpy(),
                       2 * A.matrix @ U.to_numpy()
                       + 2 * L.to_numpy() @ C @ (R.to_numpy().T @ U.to_numpy()))
