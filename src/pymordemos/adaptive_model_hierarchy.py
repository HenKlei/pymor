#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissingError
from pymor.models.hierarchy import AdaptiveModelHierarchy
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.neural_network import NeuralNetworkReductor


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

    parameter_space = fom.parameters.space((0.1, 1))

    parameters = parameter_space.sample_randomly(samples)

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

    def reduction_ml(training_data, models, reductors):
        rb_rom = models[1]
        rb_reductor = reductors[1]
        ml_reductor = reductors[0]
        error_estimator = rb_rom.error_estimator
        ml_reductor.training_set = training_data
        ml_reductor.reduced_basis = rb_reductor.bases['RB']
        ml_rom = ml_reductor.reduce(restarts=2, log_loss_frequency=10, recompute_training_data=True)
        return ml_rom.with_(error_estimator=error_estimator)

    tolerance = 1e-3

    # Settings for two-stage hierarchy
    models = [rb_reductor.reduce(), fom]
    reductors = [rb_reductor]
    reduction_methods = [reduction_rb]
    training_frequencies = [1]

    two_stage_hierarchy = AdaptiveModelHierarchy(models, reductors, reduction_methods, training_frequencies, tolerance)

    U = fom.solution_space.empty(reserve=len(parameters))
    for mu in parameters:
        u, err_est = two_stage_hierarchy.solve(mu, return_error_estimate=True)
        U.append(u)
        print(f'mu: {mu}; est. err.: {err_est}')

    print(f'Number of successful calls per model: {two_stage_hierarchy.num_successful_calls}')

    # Settings for three-stage hierarchy
    rb_reductor = CoerciveRBReductor(fom, RB=None, product=fom.energy_product,
                                     coercivity_estimator=coercivity_estimator)
    models = [None, rb_reductor.reduce(), fom]
    reductors = [ml_reductor, rb_reductor]
    reduction_methods = [reduction_ml, reduction_rb]
    training_frequencies = [20, 1]

    three_stage_hierarchy = AdaptiveModelHierarchy(models, reductors, reduction_methods, training_frequencies,
                                                   tolerance)

    U = fom.solution_space.empty(reserve=len(parameters))
    for mu in parameters:
        u, err_est = three_stage_hierarchy.solve(mu, return_error_estimate=True)
        U.append(u)
        print(f'mu: {mu}; est. err.: {err_est}')

    print(f'Number of successful calls per model: {three_stage_hierarchy.num_successful_calls}')

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
