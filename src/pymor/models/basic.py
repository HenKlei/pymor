# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from pymor.algorithms.timestepping import TimeStepper
from pymor.models.interface import Model, OutputDMuResult
from pymor.operators.constructions import ConstantOperator, IdentityOperator, VectorOperator, ZeroOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class StationaryModel(Model):
    """Generic class for models of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a vector-like right-hand side F and a (possibly non-linear) operator L.

    Note that even when solving a variational formulation where F is a
    functional and not a vector, F has to be specified as a vector-like
    |Operator| (mapping scalars to vectors). This ensures that in the complex
    case both L and F are anti-linear in the test variable.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The vector F. Either a |VectorArray| of length 1 or a vector-like
        |Operator|.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    output_d_mu_use_adjoint
        If `True`, use adjoint solution for computing output gradients
        (default behavior). See Section 1.6.2 in :cite:`HPUU09` for more
        details.
    name
        Name of the model.
    """

    def __init__(self, operator, rhs, output_functional=None, products=None,
                 error_estimator=None, visualizer=None, output_d_mu_use_adjoint=None,
                 name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), operator.source)
        if output_d_mu_use_adjoint is None:
            output_d_mu_use_adjoint = output_functional.linear and operator.linear

        assert rhs.range == operator.range
        assert rhs.source.is_scalar
        assert rhs.linear
        assert output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and output_functional.linear
        self.dim_output = output_functional.range.dim

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    dim_output:      {self.dim_output}'
        )

    def _compute(self, quantities, data, mu=None):
        if 'solution' in quantities:
            data['solution'] = self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu)
            quantities.remove('solution')

        for _, param, idx in [q for q in quantities if isinstance(q, tuple) and q[0] == 'solution_d_mu']:
            self._compute_required_quantities({'solution'}, data, mu)

            lhs_d_mu = self.operator.d_mu(param, idx).apply(data['solution'], mu=mu)
            rhs_d_mu = self.rhs.d_mu(param, idx).as_range_array(mu)
            rhs = rhs_d_mu - lhs_d_mu
            data['solution_d_mu', param, idx] = self.operator.jacobian(data['solution'], mu=mu).apply_inverse(rhs)
            quantities.remove(('solution_d_mu', param, idx))

        if 'output_d_mu' in quantities and self.output_d_mu_use_adjoint:
            if not self.operator.linear:
                raise NotImplementedError
            self._compute_required_quantities({'solution'}, data, mu)

            jacobian = self.output_functional.jacobian(data['solution'], mu)
            assert jacobian.linear
            dual_solutions = self.operator.range.empty()
            for d in range(self.output_functional.range.dim):
                dual_problem = self.with_(operator=self.operator.H,
                                          rhs=jacobian.H.as_range_array(mu)[d])
                dual_solutions.append(dual_problem.solve(mu))
            sensitivities = {}
            for (parameter, size) in self.parameters.items():
                for index in range(size):
                    output_partial_dmu = self.output_functional.d_mu(parameter, index).apply(
                        data['solution'], mu=mu).to_numpy()[0]
                    lhs_d_mu = self.operator.d_mu(parameter, index).apply2(
                        dual_solutions, data['solution'], mu=mu)[:, 0]
                    rhs_d_mu = self.rhs.d_mu(parameter, index).apply_adjoint(dual_solutions, mu=mu).to_numpy()[:, 0]
                    sensitivities[parameter, index] = (output_partial_dmu + rhs_d_mu - lhs_d_mu).reshape((1, -1))
            data['output_d_mu'] = OutputDMuResult(sensitivities)
            quantities.remove('output_d_mu')

        super()._compute(quantities, data, mu=mu)

    def deaffinize(self, arg):
        """Build |Model| with linear solution space.

        For many |Models| the solution manifold is contained in an
        affine subspace of the :attr:`~pymor.models.interface.Model.solution_space`,
        e.g. the affine space of functions with certain fixed boundary values.
        Most MOR techniques, however, construct linear approximation spaces, which
        are not fully contained in this affine subspace, even if these spaces are
        created using snapshot vectors from the subspace. Depending on the
        FOM, neglecting the affine structure of the solution space may lead to bad
        approximations or even an ill-posed ROM. A standard approach to circumvent
        this issue is to replace the FOM with an equivalent |Model| with linear
        solution space. This method can be used to obtain such a |Model|.

        Given a vector `u_0` from the affine solution space, the returned
        :class:`StationaryModel` is equivalent to solving::

            L(u(μ) + u_0, μ) = F(μ)

        When :attr:`!operator` is linear, the affine shift is
        moved to the right-hand side to obtain::

            L(u(μ), μ) = F(μ) - L(u_0, μ)

        Solutions of the original |Model| can be obtained by adding `u_0` to the
        solutions of the deaffinized |Model|.

        The :attr:`!output_functional` is adapted accordingly to
        yield the same output for given |parameter values|.

        Parameters
        ----------
        arg
            Either a |VectorArray| of length 1 containing the vector `u_0`.
            Alternatively, |parameter values| can be provided, for which the
            model is :meth:`solved <pymor.models.interface.Model.solve>` to
            obtain `u_0`.

        Returns
        -------
        The deaffinized |Model|.
        """
        if not isinstance(arg, VectorArray):
            mu = self.parameters.parse(arg)
            arg = self.solve(mu)
        assert arg in self.solution_space
        assert len(arg) == 1

        affine_shift = arg

        if self.operator.linear:
            new_operator = self.operator
            new_rhs = self.rhs - self.operator @ VectorOperator(affine_shift)
        else:
            new_operator = \
                self.operator @ (IdentityOperator(self.solution_space)
                                 + ConstantOperator(affine_shift, self.solution_space))
            new_rhs = self.rhs

        if not isinstance(self.output_functional, ZeroOperator):
            new_output_functional = \
                self.output_functional @ (IdentityOperator(self.solution_space)
                                          + ConstantOperator(affine_shift, self.solution_space))
        else:
            new_output_functional = self.output_functional

        return self.with_(operator=new_operator, rhs=new_rhs, output_functional=new_output_functional,
                          error_estimator=None)


class InstationaryModel(Model):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for given |parameter values|.
    operator
        The |Operator| L.
    rhs
        The right-hand side F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :meth:`~pymor.models.interface.Model.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 output_functional=None, products=None, error_estimator=None, visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        if isinstance(initial_data, VectorArray):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')
        mass = mass or IdentityOperator(operator.source)
        rhs = rhs or ZeroOperator(operator.source, NumpyVectorSpace(1))
        output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), operator.source)

        assert isinstance(time_stepper, TimeStepper)
        assert initial_data.source.is_scalar
        assert operator.source == initial_data.range
        assert rhs.linear
        assert rhs.range == operator.range
        assert rhs.source.is_scalar
        assert mass.linear
        assert mass.source == mass.range
        assert mass.source == operator.source
        assert output_functional.source == operator.source

        try:
            dim_input = [op.parameters['input']
                         for op in [operator, rhs, output_functional] if 'input' in op.parameters].pop()
        except IndexError:
            dim_input = 0

        super().__init__(dim_input=dim_input, products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.parameters_internal = dict(self.parameters_internal, t=1)
        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        self.dim_output = output_functional.range.dim

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    T: {self.T}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    dim_input:       {self.dim_input}\n'
            f'    dim_output:      {self.dim_output}'
        )

    def with_time_stepper(self, **kwargs):
        return self.with_(time_stepper=self.time_stepper.with_(**kwargs))

    def _compute(self, quantities, data, mu=None):
        if 'solution' in quantities:
            U0 = self.initial_data.as_range_array(mu)
            U = self.time_stepper.solve(operator=self.operator,
                                        rhs=None if isinstance(self.rhs, ZeroOperator) else self.rhs,
                                        initial_data=U0,
                                        mass=None if isinstance(self.mass, IdentityOperator) else self.mass,
                                        initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)
            data['solution'] = U
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)

    def to_lti(self):
        """Convert model to |LTIModel|.

        This method interprets the given model as an |LTIModel|
        in the following way::

            -self.operator         -> A
            self.rhs               -> B
            self.output_functional -> C
            None                   -> D
            self.mass              -> E
        """
        A = -self.operator
        B = self.rhs
        C = self.output_functional
        E = self.mass

        if not all(op.linear for op in [A, B, C, E]):
            raise ValueError('Operators not linear.')

        from pymor.models.iosys import LTIModel
        return LTIModel(A, B, C, E=E, T=self.T, time_stepper=self.time_stepper, num_values=self.num_values,
                        error_estimator=self.error_estimator, visualizer=self.visualizer, name=self.name)
