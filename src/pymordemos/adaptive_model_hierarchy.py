#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS as COLORS
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissingError
from pymor.models.hierarchy import AdaptiveModelHierarchy
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.neural_network import NeuralNetworkReductor


def draw_current_plot():
    #display.clear_output(wait=True)
    #display.display(plt.gcf())

    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()
    import time
    time.sleep(0.1)


def adaptive_hierarchy_monte_carlo(hierarchy, model_names, parameter_space, quantity_of_interest,
                                   max_num_samples=None, plotter=False):
    i = 0
    results = {'qoi': [], 'model': []}

    if plotter:
        fig, axs = plt.subplots(2, 2)

        colors = list(COLORS)

        ranges = list(parameter_space.ranges.values())
        param_dim = len(ranges)
        if param_dim in (1, 2):
            for k, name in enumerate(model_names):
                axs[0][0].plot([], [], c=colors[k], label=name)
            axs[0][0].legend(loc='upper right')
            axs[0][0].set_title('Selected parameters')
            axs[0][0].set_xlim(ranges[0][0], ranges[0][1])
            if param_dim == 2:
                axs[0][0].set_ylim(ranges[1][0], ranges[1][1])
            else:
                axs[0][0].set_ylim(-0.1, 0.1)

        axs[0][1].set_title('Quantity of interest')
        for k, name in enumerate(model_names):
            axs[0][1].plot([], [], c=colors[k], label=name)
        line_expectation, = axs[0][1].plot([], [], c=colors[-2], label='Estimated expectation')
        axs[0][1].legend(loc='upper right')

        axs[1][0].set_title('Models in the hierarchy')
        num_models = len(hierarchy.models)
        text_elements = []
        for k, name in enumerate(model_names):
            elem = axs[1][0].text((k + 1.) / (num_models + 1.), 1. - (k + 1.) / (num_models + 1.), name,
                                  ha='center', va='center', fontsize=40, color='gray',
                                  bbox=dict(boxstyle='round', edgecolor='gray', facecolor=(0.9, 0.9, .9, .5)))
            text_elements.append(elem)

        axs[1][1].set_title('Timings')
        for k, name in enumerate(model_names):
            axs[1][1].plot([], [], c=colors[k], label=name)
        axs[1][1].legend(loc='upper right')

        plt.show(block=False)

    expectations = []

    while True:
        if i == max_num_samples:
            break

        #time.sleep(1)

        for k, elem in enumerate(text_elements):
            elem.set_color('gray')
        plotter()

        mu = parameter_space.sample_randomly()
        tic = time.perf_counter()
        model_num, qoi = quantity_of_interest(hierarchy, mu)
        required_time = time.perf_counter() - tic
        # TODO: Put time measurement into compute method of the hierarchy
        # to obtain the individual timings!

        #time.sleep(1)

        results['model'].append(model_num)
        results['qoi'].append(qoi)
        if plotter:
            mu_numpy = mu.to_numpy()
            if param_dim == 1:
                mu_numpy = np.hstack([mu_numpy, 0.])
            if len(mu_numpy) == 2:
                axs[0][0].scatter(*mu_numpy, c=colors[model_num])

            axs[0][1].scatter(i, qoi, c=colors[model_num])
            if i > 0:
                expectations.append((expectations[-1]*i + qoi) / (i+1))
            else:
                expectations.append(qoi)
            line_expectation.set_xdata(np.arange(i+1))
            line_expectation.set_ydata(expectations)

            text_elements[model_num].set_color(colors[model_num])

            axs[1][1].bar(i, required_time, width=1., bottom=0., align='edge', color=colors[model_num])

            plotter()

        i += 1
    return results


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    samples: int = Argument(..., help='Number of samples to query the model hierarchy for.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
):
    """Model oder reduction with neural networks (approach by Hesthaven and Ubbiali)."""
    if not config.HAVE_TORCH:
        raise TorchMissingError

    fom, mu_bar = create_fom(fv, grid_intervals)

    def quantity_of_interest(model, mu):
        u_mu, model_num = model.solve(mu)
        # TODO: Implement this differently! Call model.output(mu)!
        # To this end: Implement output method for model hierarchy!
        return model_num, fom.output_functional.assemble(mu).apply(u_mu, mu=mu).to_numpy()

    parameter_space = fom.parameters.space((0.1, 1))

    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
    rb_reductor = CoerciveRBReductor(fom, RB=None, product=fom.energy_product,
                                     coercivity_estimator=coercivity_estimator)
    ml_reductor = NeuralNetworkReductor(fom=fom, training_set=None, validation_set=None,
                                        ann_mse=None, pod_params={'product': fom.energy_product})

    def reduction_rb(training_data, models, reductors):
        U = fom.solution_space.empty(reserve=len(training_data))
        for _, u in training_data:
            U.append(u)
        RB, _ = pod(U, product=fom.energy_product)
        reductors[0].extend_basis(RB)
        return reductors[0].reduce()

    def post_reduction_rb(training_data, models, reductors):
        return []

    tolerance = 5e-3

    # Settings for the two-stage hierarchy
    models = [rb_reductor.reduce(), fom]
    model_names = ['RB-ROM', 'FOM']
    reductors = [rb_reductor]
    reduction_methods = [reduction_rb]
    post_reduction_methods = [post_reduction_rb]
    training_frequencies = [1]

    two_stage_hierarchy = AdaptiveModelHierarchy(models, reductors, reduction_methods, post_reduction_methods,
                                                 training_frequencies, tolerance)

    max_num_samples = 100

    results = adaptive_hierarchy_monte_carlo(two_stage_hierarchy, model_names, parameter_space, quantity_of_interest,
                                             max_num_samples=max_num_samples, plotter=draw_current_plot)
    print(results)
    plt.show()

    # Settings for the three-stage hierarchy
    rb_reductor = CoerciveRBReductor(fom, RB=None, product=fom.energy_product,
                                     coercivity_estimator=coercivity_estimator)

    def reduction_ml(training_data, models, reductors):
        rb_rom = models[1]
        rb_reductor = reductors[1]
        ml_reductor = reductors[0]
        error_estimator = rb_rom.error_estimator
        ml_reductor.training_set = training_data
        ml_reductor.reduced_basis = rb_reductor.bases['RB']
        ml_rom = ml_reductor.reduce(restarts=2, log_loss_frequency=10, recompute_training_data=True)
        return ml_rom.with_(error_estimator=error_estimator)

    def post_reduction_ml(training_data, models, reductors):
        return []

    def post_reduction_rb(training_data, models, reductors):
        return [reduction_ml(training_data[1], models, reductors)]

    models = [None, rb_reductor.reduce(), fom]
    model_names = ['ML-ROM', 'RB-ROM', 'FOM']
    reductors = [ml_reductor, rb_reductor]
    reduction_methods = [reduction_ml, reduction_rb]
    post_reduction_methods = [post_reduction_ml, post_reduction_rb]
    training_frequencies = [20, 1]

    three_stage_hierarchy = AdaptiveModelHierarchy(models, reductors, reduction_methods, post_reduction_methods,
                                                   training_frequencies, tolerance)

    results = adaptive_hierarchy_monte_carlo(three_stage_hierarchy, model_names, parameter_space, quantity_of_interest,
                                             max_num_samples=max_num_samples, plotter=draw_current_plot)
    print(results)
    plt.show()

def create_fom(fv, grid_intervals):
    f = LincombFunction(
        [ExpressionFunction('10', 2), ConstantFunction(1., 2)],
        [ProjectionParameterFunctional('mu'), 0.1])
    g = LincombFunction(
        [ExpressionFunction('2 * x[0]', 2), ConstantFunction(1., 2)],
        [ProjectionParameterFunctional('mu'), 0.5])

    problem = StationaryProblem(
        domain=RectDomain(),
        rhs=f,
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[0]', 2), ExpressionFunction('x[0]', 2)],
            [ProjectionParameterFunctional('mu'), 1]),
        dirichlet_data=g,
        outputs=[('l2', f)],
        name='2DProblem'
    )
    mu_bar = problem.parameters.parse([0.5])

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(grid_intervals), mu_energy_product=mu_bar)

    return fom, mu_bar


if __name__ == '__main__':
    run(main)
