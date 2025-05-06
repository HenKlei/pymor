import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from pymor.algorithms.projection import project
from pymor.core.base import BasicObject
from pymor.models.interface import Model
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


SURROGATE_CLASSES = {'kernel': KernelRidge,
                     'nn': MLPRegressor,
                     'gpr': GaussianProcessRegressor}


class MachineLearningInstationaryReductor(BasicObject):
    def __init__(self, fom, training_set, validation_set, method='kernel', model_parameters={}, product=None):
        self.reduced_basis = None
        self.training_data = None

        if method not in SURROGATE_CLASSES:
            raise NotImplementedError(f"The machine learning method '{method}' is not available!")

        surrogate_class = SURROGATE_CLASSES[method]
        self.surrogate = surrogate_class(**model_parameters)

        self.__auto_init(locals())

    def reduce(self):
        training_inputs = np.array([mu.to_numpy() for (mu, coeffs) in self.training_data])
        training_targets = np.array([coeffs for (_, coeffs) in self.training_data])
        self.surrogate.fit(training_inputs, training_targets)
        projected_output_functional = project(self.fom.output_functional, None, self.reduced_basis)
        return MachineLearningInstationaryModel(self.fom.T, self.fom.time_stepper.nt, self.surrogate,
                                                output_functional=projected_output_functional,
                                                output_dim=training_targets.shape[1])

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert hasattr(self, 'reduced_basis')
        return self.reduced_basis.lincomb(u.to_numpy())


class MachineLearningInstationaryModel(Model):
    def __init__(self, T, nt, surrogate, output_dim=1, parameters={}, scaling_parameters={},
                 output_functional=None, products=None, error_estimator=None,
                 visualizer=None, name=None, time_stepper=None):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.solution_space = NumpyVectorSpace(output_dim)
        output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
        assert output_functional.source == self.solution_space
        self.dim_output = output_functional.range.dim
        self.__auto_init(locals())

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            # collect all inputs in a single numpy array
            inputs = np.array([mu.with_(t=t).to_numpy()
                               for t in np.linspace(0., self.T, self.nt + 1)])
            # pass batch of inputs to machine learning method
            result = self.surrogate.predict(inputs)
            result[0] = np.zeros_like(result[0])
            # convert result into element from solution space
            data['solution'] = self.solution_space.make_array(result)
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)
