# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

from pymor.models.interface import Model


class AdaptiveModelHierarchy(Model):
    def __init__(self, models, reductors, reduction_methods, post_reduction_methods, training_frequencies, tolerance,
                 name=None, visualizer=None):
        # TODO: Adaptive tolerance?!?! How to deal with training data?
        super().__init__(visualizer=visualizer, name=name)
        self.num_models = len(models)
        self.training_data = [[] for _ in range(self.num_models)]
        self.num_successful_calls = [0, ] * self.num_models
        self.num_error_estimates = [0, ] * self.num_models
        self.training_times = [0, ] * self.num_models
        self.runtimes = [0, ] * self.num_models

        assert ((self.num_models - 1) == len(reductors) == len(reduction_methods)
                == len(training_frequencies))
        assert tolerance >= 0.

        self.dim_output = models[-1].dim_output
        self.output_functional = models[-1].output_functional

        self.len_previous_training_data = [0, ] * self.num_models

        self.__auto_init(locals())
        self.set_tolerance(tolerance)

    def reconstruct(self, u, i):
        try:
            U = self.reductors[i].reconstruct(u)
        except IndexError:
            U = u
        return U

    def set_tolerance(self, tol):
        self._tolerance = tol

    def get_tolerance(self):
        return self._tolerance

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            for i, model in enumerate(self.models):
                if model is None:
                    continue

                if i != self.num_models - 1:
                    tic = time.perf_counter()
                    est_err = model.estimate_error(mu=mu)
                    self.runtimes[i] += time.perf_counter() - tic
                    self.num_error_estimates[i] += 1
                else:
                    est_err = -1

                if est_err <= self.get_tolerance():
                    self.num_successful_calls[i] += 1
                    tic = time.perf_counter()
                    sol = model.solve(mu=mu)
                    toc = time.perf_counter()
                    self.runtimes[i] += toc - tic
                    data['solution_time'] = toc - tic
                    # TODO: Avoid necessity of reconstruction!!!
                    # If doing so: Make sure to provide information about model that produced
                    # the result and about the reductor that can be used for reconstruction!!!
                    # Make sure to adjust handling of training data in NeuralNetworkReductor
                    # such that it can also deal with reduced coefficients as data!!!
                    # TODO: How to deal with reduced basis extension?!?!
                    # Adjustment of training data necessary (zero padding;
                    # removal of previous training data;
                    # new machine learning surrogate for additional coefficients; etc.)!!!
                    data['solution'] = sol
                    model_number = i
                    data['model_number'] = model_number
                    if i > 0:
                        self.len_previous_training_data[i-1] = len(self.training_data[i-1])
                        self.training_data[i-1].append((mu, sol))
                    quantities.remove('solution')
                    if 'solution_error_estimate' in quantities:
                        if est_err == -1:
                            est_err = None
                        data['solution_error_estimate'] = est_err
                        quantities.remove('solution_error_estimate')
                    break

            # Perform training of reduced models
            for j, (freq, model, red_meth) in enumerate(zip(self.training_frequencies, self.models,
                                                            self.reduction_methods)):
                if j < model_number and len(self.training_data[j]) > 0 and len(self.training_data[j]) % freq == 0:
                    tic = time.perf_counter()
                    self.models[j] = red_meth(self.training_data[j], self.len_previous_training_data[j],
                                              self.models[j:], self.reductors[j:])
                    self.models[:j] = self.post_reduction_methods[j](self.training_data[:j+1], self.models[:j+1],
                                                                     self.reductors[:j+1])
                    toc = time.perf_counter()
                    self.training_times[j] += toc - tic
            data['training_time'] = toc - tic

        if 'output' in quantities:
            data['output'] = self.models[model_number].compute(data={'solution': sol}, output=True, mu=mu)['output']
            quantities.remove('output')

        super()._compute(quantities, data, mu=mu)
