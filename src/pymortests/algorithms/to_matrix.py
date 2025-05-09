# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.to_matrix import to_matrix
from pymor.core.config import config
from pymor.operators.block import BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import (
    AdjointOperator,
    ComponentProjectionOperator,
    IdentityOperator,
    LowRankOperator,
    LowRankUpdatedOperator,
    VectorArrayOperator,
    ZeroOperator,
)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def assert_type_and_allclose(A, Aop, default_format):
    if default_format == 'dense':
        assert isinstance(to_matrix(Aop), np.ndarray)
        assert np.allclose(A, to_matrix(Aop))
    elif default_format == 'sparse':
        assert sps.issparse(to_matrix(Aop))
        assert np.allclose(A, to_matrix(Aop).toarray())
    else:
        assert getattr(sps, 'isspmatrix_' + default_format)(to_matrix(Aop))
        assert np.allclose(A, to_matrix(Aop).toarray())

    assert isinstance(to_matrix(Aop, format='dense'), np.ndarray)
    assert np.allclose(A, to_matrix(Aop, format='dense'))

    assert sps.isspmatrix_csr(to_matrix(Aop, format='csr'))
    assert np.allclose(A, to_matrix(Aop, format='csr').toarray())


@pytest.mark.builtin
def test_to_matrix_NumpyMatrixOperator(rng):
    A = rng.standard_normal((2, 2))

    Aop = NumpyMatrixOperator(A)
    assert_type_and_allclose(A, Aop, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    assert_type_and_allclose(A, Aop, 'csc')


@pytest.mark.builtin
def test_to_matrix_BlockOperator(rng):
    A11 = rng.standard_normal((2, 2))
    A12 = rng.standard_normal((2, 3))
    A21 = rng.standard_normal((3, 2))
    A22 = rng.standard_normal((3, 3))
    B = np.block([[A11, A12], [A21, A22]])

    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A21op = NumpyMatrixOperator(A21)
    A22op = NumpyMatrixOperator(A22)
    Bop = BlockOperator([[A11op, A12op], [A21op, A22op]])
    assert_type_and_allclose(B, Bop, 'dense')

    A11op = NumpyMatrixOperator(sps.csc_matrix(A11))
    A12op = NumpyMatrixOperator(A12)
    A21op = NumpyMatrixOperator(A21)
    A22op = NumpyMatrixOperator(A22)
    Bop = BlockOperator([[A11op, A12op], [A21op, A22op]])
    assert_type_and_allclose(B, Bop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_BlockDiagonalOperator(rng):
    A1 = rng.standard_normal((2, 2))
    A2 = rng.standard_normal((3, 3))
    B = np.block([[A1, np.zeros((2, 3))],
                  [np.zeros((3, 2)), A2]])

    A1op = NumpyMatrixOperator(A1)
    A2op = NumpyMatrixOperator(A2)
    Bop = BlockDiagonalOperator([A1op, A2op])
    assert_type_and_allclose(B, Bop, 'sparse')

    A1op = NumpyMatrixOperator(sps.csc_matrix(A1))
    A2op = NumpyMatrixOperator(A2)
    Bop = BlockDiagonalOperator([A1op, A2op])
    assert_type_and_allclose(B, Bop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_AdjointOperator(rng):
    A = rng.standard_normal((2, 2))
    S = rng.standard_normal((2, 2))
    S = S.dot(S.T)
    R = rng.standard_normal((2, 2))
    R = R.dot(R.T)

    Aop = NumpyMatrixOperator(A)
    Aadj = AdjointOperator(Aop)
    assert_type_and_allclose(A.T, Aadj, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Aadj = AdjointOperator(Aop)
    assert_type_and_allclose(A.T, Aadj, 'sparse')

    Aop = NumpyMatrixOperator(A)
    Sop = NumpyMatrixOperator(S)
    Rop = NumpyMatrixOperator(R)
    Aadj = AdjointOperator(Aop, source_product=Sop, range_product=Rop)
    assert_type_and_allclose(spla.solve(S, A.T.dot(R)), Aadj, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Sop = NumpyMatrixOperator(S)
    Rop = NumpyMatrixOperator(R)
    Aadj = AdjointOperator(Aop, source_product=Sop, range_product=Rop)
    assert_type_and_allclose(spla.solve(S, A.T.dot(R)), Aadj, 'dense')

    Aop = NumpyMatrixOperator(A)
    Sop = NumpyMatrixOperator(S)
    Rop = NumpyMatrixOperator(sps.csc_matrix(R))
    Aadj = AdjointOperator(Aop, source_product=Sop, range_product=Rop)
    assert_type_and_allclose(spla.solve(S, A.T.dot(R)), Aadj, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Sop = NumpyMatrixOperator(sps.csc_matrix(S))
    Rop = NumpyMatrixOperator(sps.csc_matrix(R))
    Aadj = AdjointOperator(Aop, source_product=Sop, range_product=Rop)
    assert_type_and_allclose(spla.solve(S, A.T.dot(R)), Aadj, 'sparse')


@pytest.mark.builtin
def test_to_matrix_ComponentProjectionOperator():
    dofs = np.array([0, 1, 2, 4, 8])
    n = 10
    A = np.zeros((len(dofs), n))
    A[range(len(dofs)), dofs] = 1

    source = NumpyVectorSpace(n)
    Aop = ComponentProjectionOperator(dofs, source)
    assert_type_and_allclose(A, Aop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_ConcatenationOperator(rng):
    A = rng.standard_normal((2, 3))
    B = rng.standard_normal((3, 4))
    C = A.dot(B)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = Aop @ Bop
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Bop = NumpyMatrixOperator(B)
    Cop = Aop @ Bop
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(sps.csc_matrix(B))
    Cop = Aop @ Bop
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Bop = NumpyMatrixOperator(sps.csc_matrix(B))
    Cop = Aop @ Bop
    assert_type_and_allclose(A, Aop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_IdentityOperator():
    n = 3
    I = np.eye(n)

    Iop = IdentityOperator(NumpyVectorSpace(n))
    assert_type_and_allclose(I, Iop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_LincombOperator(rng):
    A = rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 2))
    a = rng.standard_normal()
    b = rng.standard_normal()
    C = a * A + b * B.dot(B.T)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = Aop * a + (Bop @ Bop.H) * b
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Bop = NumpyMatrixOperator(B)
    Cop = Aop * a + (Bop @ Bop.H) * b
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(sps.csc_matrix(B))
    Cop = Aop * a + (Bop @ Bop.H) * b
    assert_type_and_allclose(C, Cop, 'dense')

    Aop = NumpyMatrixOperator(sps.csc_matrix(A))
    Bop = NumpyMatrixOperator(sps.csc_matrix(B))
    Cop = Aop * a + (Bop @ Bop.H) * b
    assert_type_and_allclose(C, Cop, 'sparse')


@pytest.mark.builtin
def test_to_matrix_LowRankOperator(rng):
    m = 6
    n = 5
    r = 2
    L = rng.standard_normal((m, r))
    Lva = NumpyVectorSpace.make_array(L)
    C = rng.standard_normal((r, r))
    R = rng.standard_normal((n, r))
    Rva = NumpyVectorSpace.make_array(R)

    LR = LowRankOperator(Lva, C, Rva)
    assert_type_and_allclose(L @ C @ R.T, LR, 'dense')

    LR = LowRankOperator(Lva, C, Rva, inverted=True)
    assert_type_and_allclose(L @ spla.solve(C, R.T), LR, 'dense')


@pytest.mark.builtin
def test_to_matrix_LowRankUpdatedOperator(rng):
    m = 6
    n = 5
    r = 2
    A = rng.standard_normal((m, n))
    Aop = NumpyMatrixOperator(A)
    L = rng.standard_normal((m, r))
    Lva = NumpyVectorSpace.make_array(L)
    C = rng.standard_normal((r, r))
    R = rng.standard_normal((n, r))
    Rva = NumpyVectorSpace.make_array(R)
    LR = LowRankOperator(Lva, C, Rva)

    op = LowRankUpdatedOperator(Aop, LR, 1, 1)
    assert_type_and_allclose(A + L @ C @ R.T, op, 'dense')


@pytest.mark.builtin
def test_to_matrix_VectorArrayOperator(rng):
    V = rng.standard_normal((10, 2))

    Vva = NumpyVectorSpace.make_array(V)
    Vop = VectorArrayOperator(Vva)
    assert_type_and_allclose(V, Vop, 'dense')

    Vop = VectorArrayOperator(Vva, adjoint=True)
    assert_type_and_allclose(V.T, Vop, 'dense')


@pytest.mark.builtin
def test_to_matrix_ZeroOperator():
    n = 3
    m = 4
    Z = np.zeros((n, m))

    Zop = ZeroOperator(NumpyVectorSpace(n), NumpyVectorSpace(m))
    assert_type_and_allclose(Z, Zop, 'sparse')


if config.HAVE_DUNEGDT:
    from dune.xt.la import IstlSparseMatrix, SparsityPatternDefault

    from pymor.bindings.dunegdt import DuneXTMatrixOperator

    def test_to_matrix_DuneXTMatrixOperator(rng):
        A = rng.standard_normal((2, 2))

        pattern = SparsityPatternDefault(2)
        for ii in range(2):
            for jj in range(2):
                pattern.insert(ii, jj)
        pattern.sort()
        mat = IstlSparseMatrix(2, 2, pattern)
        for ii in range(2):
            for jj in range(2):
                mat.set_entry(ii, jj, A[ii][jj])
        Aop = DuneXTMatrixOperator(mat)

        assert_type_and_allclose(A, Aop, 'sparse')
