# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.models.interface import Model


class AdaptiveModelHierarchy(Model):
    def __init__(self, models, reductors, reduction_methods, post_reduction_methods, training_frequencies, tolerance,
                 name=None, visualizer=None):
        # TODO: Adaptive tolerance?!?! How to deal with training data?
        super().__init__(visualizer=visualizer, name=name)
        self.num_models = len(models)
        self.training_data = [[] for _ in range(self.num_models)]
        self.num_successful_calls = [0, ] * self.num_models

        assert ((self.num_models - 1) == len(reductors) == len(reduction_methods)
                == len(training_frequencies))
        assert tolerance > 0.

        self.dim_output = models[-1].dim_output
        self.output_functional = models[-1].output_functional

        self.__auto_init(locals())

    def reconstruct(self, u, i):
        return self.reductors[i].reconstruct(u)

    def _compute(self, quantities, data, mu):
        print(mu)
        if 'solution' in quantities:
            for i, model in enumerate(self.models):
                if model is None:
                    continue

                if i != self.num_models - 1:
                    est_err = model.estimate_error(mu=mu)
                    rec_meth = self.reductors[i].reconstruct
                else:
                    est_err = -1

                    def rec_meth(x):
                        return x

                if est_err <= self.tolerance:
                    self.num_successful_calls[i] += 1
                    sol = model.solve(mu=mu)
                    # TODO: Avoid necessity of reconstruction!!!
                    # If doing so: Make sure to provide information about model that produced
                    # the result and about the reductor that can be used for reconstruction!!!
                    # Make sure to adjust handling of training data in NeuralNetworkReductor
                    # such that it can also deal with reduced coefficients as data!!!
                    # TODO: How to deal with reduced basis extension?!?!
                    # Adjustment of training data necessary (zero padding;
                    # removal of previous training data;
                    # new machine learning surrogate for additional coefficients; etc.)!!!
                    sol = rec_meth(sol)
                    data['solution'] = sol
                    data['model_number'] = i
                    if i > 0:
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
                if j < i and len(self.training_data[j]) > 0 and len(self.training_data[j]) % freq == 0:
                    self.models[j] = red_meth(self.training_data[j], self.models[j:], self.reductors[j:])
                    self.models[:j] = self.post_reduction_methods[j](self.training_data[:j+1], self.models[:j+1],
                                                                     self.reductors[:j+1])

        super()._compute(quantities, data, mu=mu)
