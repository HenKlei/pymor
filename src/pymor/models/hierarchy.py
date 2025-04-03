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
        model_number = -1
        data['error_estimates'] = []
        data['runtimes'] = [0, ] * self.num_models
        data['training_times'] = [0, ] * self.num_models

        if 'solution' in quantities:
            for i, model in enumerate(self.models):
                if model is None:
                    data['error_estimates'].append(None)
                    continue

                if i != self.num_models - 1:
                    tic = time.perf_counter()
                    est_err = model.estimate_error(mu=mu)
                    if hasattr(est_err, '__len__') and len(est_err) > 0:
                        est_err = est_err[0]
                    runtime = time.perf_counter() - tic
                    self.runtimes[i] += runtime
                    data['runtimes'][i] = runtime
                    self.num_error_estimates[i] += 1
                    data['error_estimates'].append(est_err)
                else:
                    est_err = -1
                    data['runtimes'][i] = 0.

                if est_err <= self.get_tolerance():
                    self.num_successful_calls[i] += 1
                    tic = time.perf_counter()
                    sol = model.solve(mu=mu)
                    toc = time.perf_counter()
                    runtime = toc - tic
                    self.runtimes[i] += runtime
                    data['solution_time'] = runtime
                    data['runtimes'][i] += runtime
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

        if 'output' in quantities:
            if model_number == -1:
                for i, model in enumerate(self.models):
                    if model is None:
                        data['error_estimates'].append(None)
                        continue

                    if i != self.num_models - 1:
                        tic = time.perf_counter()
                        est_err = model.estimate_output_error(mu=mu)#[0]
                        runtime = time.perf_counter() - tic
                        self.runtimes[i] += runtime
                        data['runtimes'][i] = runtime
                        self.num_error_estimates[i] += 1
                        data['error_estimates'].append(est_err)
                    else:
                        est_err = -1
                        data['runtimes'][i] = 0.

                    if est_err <= self.get_tolerance():
                        self.num_successful_calls[i] += 1
                        model_number = i
                        data['model_number'] = model_number
                        break

            tic = time.perf_counter()
            sol_data = self.models[model_number].compute(solution=True, output=True, mu=mu)
            runtime = time.perf_counter() - tic
            data['output'] = sol_data['output']
            if model_number > 0:
                self.len_previous_training_data[model_number-1] = len(self.training_data[model_number-1])
                self.training_data[model_number-1].append((mu, sol_data['solution']))
            self.runtimes[model_number] += runtime
            data['runtimes'][model_number] += runtime
            quantities.remove('output')

        # Perform training of reduced models
        for j, (freq, model, red_meth) in enumerate(zip(self.training_frequencies, self.models,
                                                        self.reduction_methods)):
            if j < model_number and len(self.training_data[j]) > 0 and len(self.training_data[j]) % freq == 0:
                tic = time.perf_counter()
                self.models[j] = red_meth(self.training_data[j], self.len_previous_training_data[j],
                                          self.models[j:], self.reductors[j:])
                self.models[:j] = self.post_reduction_methods[j](self.training_data[:j+1], self.models[:j+1],
                                                                 self.reductors[:j+1])
                training_time = time.perf_counter() - tic
                self.training_times[j] += training_time
                data['training_times'][j] += training_time

        super()._compute(quantities, data, mu=mu)
