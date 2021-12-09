#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
import numpy as np
from typer import Argument, run
import matplotlib.pyplot as plt

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissing
from pymor.reductors.neural_network import (NeuralNetworkInstationaryReductor,
                                            NeuralNetworkInstationaryStatefreeOutputReductor)


def main(
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),
    validation_samples: int = Argument(..., help='Number of samples used for validation during the training phase.'),
    fom_number: int = Argument(..., help='Selects FOMs [0, 1] with scalar or vector valued outputs.')
):
    """Model oder reduction with neural networks for an instationary problem

    Using the approach by Hesthaven and Ubbiali.
    """
    if not config.HAVE_TORCH:
        raise TorchMissing()

    assert fom_number in [0, 1], f'No FOM available for fom_number {fom_number}'

    fom = create_fom(fom_number)

    parameter_space = fom.parameters.space(1., 100.)

    training_set = parameter_space.sample_uniformly(training_samples)
    validation_set = parameter_space.sample_randomly(validation_samples)

    basis_size = 10

    ann_reductor = NeuralNetworkInstationaryReductor(fom, training_set, validation_set,
                                                     pod_params={'product': fom.h1_0_semi_product},
                                                     basis_size=basis_size, ann_mse=None,
                                                     scale_inputs=True, scale_outputs=True)
    ann_rom = ann_reductor.reduce(hidden_layers='[30, 30, 30]', restarts=5)

    coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
    reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)
    reductor.extend_basis(ann_reductor.reduced_basis, method='trivial')
    rom = reductor.reduce()

    test_set = parameter_space.sample_randomly(10)

    speedups_ann = []
    speedups_rom = []

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.solution_space.empty(reserve=len(test_set))
    U_ann_red = fom.solution_space.empty(reserve=len(test_set))
    U_rom = fom.solution_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.time()
        U.append(fom.solve(mu))
        time_fom = time.time() - tic

        tic = time.time()
        U_ann_red.append(ann_reductor.reconstruct(ann_rom.solve(mu)))
        time_ann_red = time.time() - tic

        speedups_ann.append(time_fom / time_ann_red)

        tic = time.time()
        U_rom.append(reductor.reconstruct(rom.solve(mu)))
        time_rom = time.time() - tic

        speedups_rom.append(time_fom / time_rom)

    absolute_errors_ann = (U - U_ann_red).norm2()
    relative_errors_ann = (U - U_ann_red).norm2() / U.norm2()

    absolute_errors_rom = (U - U_rom).norm2()
    relative_errors_rom = (U - U_rom).norm2() / U.norm2()

    output_reductor = NeuralNetworkInstationaryStatefreeOutputReductor(fom, training_set, validation_set,
                                                                       validation_loss=None)
    output_rom = output_reductor.reduce(restarts=5)

    outputs = []
    outputs_red = []
    outputs_speedups = []

    print(f'Performing test on set of size {len(test_set)} ...')

    for mu in test_set:
        tic = time.perf_counter()
        outputs.append(fom.compute(output=True, mu=mu)['output'])
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red.append(output_rom.compute(output=True, mu=mu)['output'])
        time_red = time.perf_counter() - tic

        outputs_speedups.append(time_fom / time_red)

    outputs = np.squeeze(np.array(outputs))
    outputs_red = np.squeeze(np.array(outputs_red))

    outputs_absolute_errors = np.abs(outputs - outputs_red)
    outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

    print('Results for state approximation using ANNs:')
    print(f'Average absolute error: {np.average(absolute_errors_ann)}')
    print(f'Average relative error: {np.average(relative_errors_ann)}')
    print(f'Median of speedup: {np.median(speedups_ann)}')

    print('Results for state approximation using POD-Galerkin-ROM:')
    print(f'Average absolute error: {np.average(absolute_errors_rom)}')
    print(f'Average relative error: {np.average(relative_errors_rom)}')
    print(f'Median of speedup: {np.median(speedups_rom)}')

    print()
    print('Results for output approximation using ANNs:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors)}')
    print(f'Average relative error: {np.average(outputs_relative_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')

    mu = parameter_space.sample_randomly(1)[0]
    U = fom.solve(mu)
    U_ANN = ann_reductor.reconstruct(ann_rom.solve(mu))
    U_ROM = reductor.reconstruct(rom.solve(mu))
    fom.visualize((U, U_ANN, U_ROM, U - U_ANN, U - U_ROM), legend=('Detailed Solution', 'Reduced Solution using ANNs', 'Reduced Solution using POD-Galerkin-ROM', 'Error using ANNs', 'Error using POD-Galerkin ROM'),
                  separate_colorbars=True)

    for i in range(basis_size):
        plt.figure(i)
        plt.plot(np.linspace(0., fom.T, len(U)), ann_reductor.reduced_basis.inner(U)[i])
        plt.plot(np.linspace(0., fom.T, len(U)), ann_rom.solve(mu).to_numpy()[..., i])
        plt.plot(np.linspace(0., fom.T, len(U)), rom.solve(mu).to_numpy()[..., i])
        plt.legend(['orthogonal projection', 'ANN-ROM', 'POD-Galerkin-ROM'])
        plt.title(f"POD Mode {i}")
        plt.show()

    o_fom = fom.compute(output=True, mu=mu)['output']
    o_rom = output_rom.compute(output=True, mu=mu)['output']

    for i in range(o_fom.shape[1]):
        plt.figure(i + basis_size)
        plt.plot(np.linspace(0., fom.T, len(o_fom)), o_fom[:, i])
        plt.plot(np.linspace(0., fom.T, len(o_fom)), o_rom[:, i])
        plt.legend(['orthogonal projection', 'ANN-ROM'])
        plt.title(f"Output component {i}")
        plt.show()


def create_fom(fom_number):
    from pymordemos.parabolic_mor import discretize_pymor
    fom = discretize_pymor()

    if fom_number == 0:
        fom = fom.with_(output_functional=fom.rhs.operators[0].H)
    else:
        random_matrix_1 = np.random.rand(2, fom.solution_space.dim)
        op = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
        fom = fom.with_(output_functional=op)

    return fom


if __name__ == '__main__':
    run(main)
