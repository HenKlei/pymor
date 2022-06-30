#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
import numpy as np
from typer import Argument, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissing
from pymor.reductors.neural_network import (NeuralNetworkInstationaryReductor,
                                            NeuralNetworkInstationaryStatefreeOutputReductor,
                                            NeuralNetworkLSTMInstationaryReductor,
                                            NeuralNetworkLSTMInstationaryStatefreeOutputReductor)


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    time_steps: int = Argument(..., help='Number of time steps used for discretization.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),
    validation_samples: int = Argument(..., help='Number of samples used for validation during the training phase.'),
):
    """Model oder reduction with neural networks for an instationary problem

    Using the approach by Hesthaven and Ubbiali and long short-term memory networks.
    """
    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(grid_intervals, time_steps)

    parameter_space = fom.parameters.space(1., 50.)

    training_set = parameter_space.sample_uniformly(training_samples)
    validation_set = parameter_space.sample_randomly(validation_samples)
    test_set = parameter_space.sample_randomly(10)

    def compute_errors_state(rom, reductor):
        speedups = []

        print(f'Performing test on set of size {len(test_set)} ...')

        U = fom.solution_space.empty(reserve=len(test_set))
        U_red = fom.solution_space.empty(reserve=len(test_set))

        for mu in test_set:
            tic = time.time()
            U.append(fom.solve(mu)[1:])
            time_fom = time.time() - tic

            tic = time.time()
            U_red.append(reductor.reconstruct(rom.solve(mu))[1:])
            time_red = time.time() - tic

            speedups.append(time_fom / time_red)

        absolute_errors = (U - U_red).norm2()
        relative_errors = (U - U_red).norm2() / U.norm2()

        return absolute_errors, relative_errors, speedups

    reductor = NeuralNetworkInstationaryReductor(fom, training_set, validation_set, basis_size=10, scale_outputs=True)
    rom = reductor.reduce(hidden_layers='[30, 30, 30]', restarts=10)
    abs_errors, rel_errors, speedups = compute_errors_state(rom, reductor)

    reductor_lstm = NeuralNetworkLSTMInstationaryReductor(fom, training_set, validation_set, basis_size=10,
                                                          scale_inputs=True, scale_outputs=True, ann_mse=None)
    rom_lstm = reductor_lstm.reduce(restarts=10)
    abs_errors_lstm, rel_errors_lstm, speedups_lstm = compute_errors_state(rom_lstm, reductor_lstm)

    def compute_errors_output(output_rom):
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

        return outputs_absolute_errors, outputs_relative_errors, outputs_speedups

    output_reductor = NeuralNetworkInstationaryStatefreeOutputReductor(fom, time_steps+1, training_set,
                                                                       validation_set, validation_loss=1e-5)
    output_rom = output_reductor.reduce(restarts=100)

    outputs_abs_errors, outputs_rel_errors, outputs_speedups = compute_errors_output(output_rom)

    output_reductor_lstm = NeuralNetworkLSTMInstationaryStatefreeOutputReductor(fom, time_steps+1, training_set,
                                                                                validation_set, validation_loss=1e-4)
    output_rom_lstm = output_reductor_lstm.reduce(restarts=10, number_layers=3, hidden_dimension=30,
                                                  learning_rate=0.1, log_loss_frequency=3)

    outputs_abs_errors_lstm, outputs_rel_errors_lstm, outputs_speedups_lstm = compute_errors_output(output_rom_lstm)

    print()
    print('Approach by Hesthaven and Ubbiali using feedforward ANNs:')
    print('=========================================================')

    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(abs_errors)}')
    print(f'Average relative error: {np.average(rel_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')

    print()
    print('Results for output approximation:')
    print(f'Average absolute error: {np.average(outputs_abs_errors)}')
    print(f'Average relative error: {np.average(outputs_rel_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')

    print()
    print()
    print('Approach using long short-term memory ANNs:')
    print('===========================================')

    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(abs_errors_lstm)}')
    print(f'Average relative error: {np.average(rel_errors_lstm)}')
    print(f'Median of speedup: {np.median(speedups_lstm)}')

    print()
    print('Results for output approximation:')
    print(f'Average absolute error: {np.average(outputs_abs_errors_lstm)}')
    print(f'Average relative error: {np.average(outputs_rel_errors_lstm)}')
    print(f'Median of speedup: {np.median(outputs_speedups_lstm)}')


def create_fom(grid_intervals, time_steps):
    print('Discretize ...')
    from fenics_navier_stokes import discretize
    fom, _ = discretize(grid_intervals, time_steps)

    return fom


if __name__ == '__main__':
    run(main)
