import os
import sys

# +
from tempfile import NamedTemporaryFile

import numpy as np
from dune.gdt import FiniteVolumeSpace, default_interpolation
from dune.xt.functions import GridFunction as GF  # noqa: N817
from dune.xt.functions import IndicatorFunction__2d_to_1x1 as IndicatorFunction
from dune.xt.functions import Spe10Model1Function
from dune.xt.grid import (
    Cube,
    Dim,
    DirichletBoundary,
    FunctionBasedBoundaryInfo,
    GridProvider2dCubeYaspgrid,
    NeumannBoundary,
    make_cube_grid,
)

from pymor.algorithms.to_matrix import to_matrix
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import BitmapFunction, ConstantFunction, ExpressionFunction, GenericFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.core.base import ImmutableObject
from pymor.discretizers.builtin.cg import discretize_instationary_cg as discretize_instationary_cg_pymor
from pymor.discretizers.builtin.grids.boundaryinfos import GenericBoundaryInfo
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.dunegdt.cg import _discretize_instationary_cg_dune as discretize_instationary_cg_dune
from pymor.discretizers.dunegdt.functions import DuneFunction, DuneGridFunction, LincombDuneGridFunction
from pymor.discretizers.dunegdt.problems import InstationaryDuneProblem, StationaryDuneProblem
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters, ParameterSpace
from pymor.parameters.functionals import (
    ConstantParameterFunctional,
    GenericParameterFunctional,
    MinThetaParameterFunctional,
    ParameterFunctional,
    ProjectionParameterFunctional,
)
from pymor.tools.floatcmp import float_cmp

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # tools.py
#from tools import simplify, to_numpy

def simplify(op):
    assert isinstance(op, LincombOperator)
    parametric_ops = []
    parametric_coeffs = []
    nonparametric_ops = []
    nonparametric_coeffs = []
    for oo, cc in zip(op.operators, op.coefficients):
        if isinstance(cc, ParameterFunctional):
            parametric_ops.append(oo)
            parametric_coeffs.append(cc)
        else:
            nonparametric_ops.append(oo)
            nonparametric_coeffs.append(cc)
    if len(nonparametric_ops) == 0:
        return op
    else:
        return op.with_(
            operators=parametric_ops + [LincombOperator(nonparametric_ops, nonparametric_coeffs).assemble()],
            coefficients=parametric_coeffs + [1,])

class ConvertedVisualizer():#ImmutableObject):

    def __init__(self, visualizer, vector_space, num_grid_elements):
        self.visualizer = visualizer
        self.vector_space = vector_space
        self.num_grid_elements = num_grid_elements
        #self.__auto_init(locals())

    def visualize(self, U, fig_width=10, fig_height=3, extent=(0, 1, 0, 1), *args, **kwargs):
        assert len(U) == 1
        v = U[0].to_numpy()
        v = v.reshape(self.num_grid_elements[1]+1, self.num_grid_elements[0]+1)
        v = np.array(np.flip(v, 0))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        fig.canvas.header_visible = False
        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        im = ax.imshow(v.reshape(self.num_grid_elements[1]+1, self.num_grid_elements[0]+1), extent=extent)
        self.im = im
        widget = fig.canvas
        return widget
        #self.visualizer.visualize(V, *args, **kwargs)

    def set(self, U):
        assert len(U) == 1
        v = U[0].to_numpy()
        v = v.reshape(self.num_grid_elements[1]+1, self.num_grid_elements[0]+1)
        v = np.array(np.flip(v, 0))
        self.im.set_data(v)

    def reset(self):
        self.im.set_data([[]])

def to_numpy(obj, **kwargs):
    if isinstance(obj, (DuneXTMatrixOperator, VectorArrayOperator)):
        return NumpyMatrixOperator(to_matrix(obj))
    elif isinstance(obj, LincombOperator):
        return obj.with_(
            operators=[to_numpy(op) for op in obj.operators],
            solver_options=None,
        )
    elif isinstance(obj, InstationaryModel):
        return obj.with_(
            operator=to_numpy(obj.operator),
            rhs=to_numpy(obj.rhs),
            mass=to_numpy(obj.mass),
            products={kk: to_numpy(vv) for kk, vv in obj.products.items()},
            initial_data=to_numpy(obj.initial_data),
            output_functional=to_numpy(obj.output_functional),
            visualizer=ConvertedVisualizer(obj.visualizer, obj.solution_space, kwargs['num_grid_elements'])
        )

    assert False, 'We should not get here!'


def dune_function_to_pymor(func, grid, num_grid_elements, bounding_box):
    assert DuneFunction.is_base_of(func) or DuneGridFunction.is_base_of(func)
    assert func.dim_domain == 2
    assert func.dim_range == 1
    assert isinstance(grid, GridProvider2dCubeYaspgrid)
    assert len(num_grid_elements) == 2
    # obtain piecewise constant interpolation
    fv_space = FiniteVolumeSpace(grid)
    df = default_interpolation(GF(func), fv_space)
    dofs = np.array(df.dofs.vector, copy=False).reshape(num_grid_elements[1], num_grid_elements[0])
    min_max = [np.min(dofs), np.max(dofs)]
    # write to png, see https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    from PIL import Image
    dofs = np.array(np.flip(dofs, 0))
    # - normalize and rescale to 0-255 range
    dofs *= (255/np.max(dofs))
    # - convert to integers
    dofs = np.uint8(dofs)
    # - create image
    filename = NamedTemporaryFile(suffix='.png', delete=False).name
    im = Image.fromarray(dofs)
    im.save(filename)
    # create pyMOR function
    return BitmapFunction.from_file(filename, bounding_box=bounding_box, range=min_max)


# -

def make_problem(regime='diffusion dominated', num_global_refines=0, spe10_perm_max=1, channel_initially_filled=False):
    """Creates problem.

    Parameters
    ----------
    regime

    spe10_perm_max
        The original range of the Spe10 Model1 function is [1e-3, 998.915], and will be (linearly)
        scaled to [1e-3, spe10_perm_max].
    channel_initially_filled

    Returns
    -------
    problem, parameter_space, mu_bar
    """
    parameter_ranges = {
        'diffusion dominated': ParameterSpace(Parameters({'Da': 1, 'Pe': 1}), {'Da': (1e-2, 10), 'Pe': (9, 11)}),
        'convection dominated': ParameterSpace(Parameters({'Da': 1, 'Pe': 1}), {'Da': (1, 10), 'Pe': (50, 100)}),
        'reaction dominated': ParameterSpace(Parameters({'Da': 1, 'Pe': 1}), {'Da': (10, 100), 'Pe': (10, 25)}),
    }

    T_end = {
        'diffusion dominated': 5,
        'convection dominated': 0.3,
        'reaction dominated': 0.7,
    }

    domain = ([0, 0], [2.5, 1])
    refine_factor = 2**num_global_refines
    num_grid_elements = [100*refine_factor, 20*refine_factor]

    grid = {
        'dune': make_cube_grid(Dim(2), Cube(), domain[0], domain[1], num_grid_elements),
        'pymor': RectGrid(num_intervals=num_grid_elements, domain=domain),
    }

    spe10_perm = {
        'dune': Spe10Model1Function(
            grid['dune'],
            filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'perm_case1.dat'),
            lower_left=domain[0],
            upper_right=domain[1],
            max=spe10_perm_max)}
    spe10_perm['pymor'] = dune_function_to_pymor(spe10_perm['dune'], grid['dune'], num_grid_elements, domain)

    BL = 1e-7
    HW = 0.34
    channel = {
        'dune': GF(grid['dune'], GF(grid['dune'], IndicatorFunction([([[0, 5], [HW, 1]], [1.]),]))),
        'pymor': ExpressionFunction(
            f'(0 <= x[0])*1.*(x[0] <= 5)*({HW} <= x[1])*(x[1] <= 1)',
            dim_domain=2)}
    washcoat = {
        'dune': GF(grid['dune'], IndicatorFunction([([[0, domain[1][0]], [0, HW]], [1.]),])),
        'pymor': ExpressionFunction(
            f'(0 <= x[0])*1.*(x[0] <= 5)*(0 <= x[1])*(x[1] <= {HW})',
            dim_domain=2)}
    inflow = {
        'dune': GF(grid['dune'], GF(grid['dune'], IndicatorFunction([([[0-BL, 0+BL], [HW, 1]], [1.]),]))),
        'pymor': ExpressionFunction(
            f'(0-{BL} <= x[0])*1.*(x[0] <= 0+{BL})*({HW} <= x[1])*(x[1] <= 1)',
            dim_domain=2)}
    outflow = {
        'dune': GF(grid['dune'], GF(grid['dune'], IndicatorFunction([([[domain[1][0]-BL, domain[1][0]+BL], [HW, 1]], [1.]),]))),
        'pymor': ExpressionFunction(
            f'(5-{BL} <= x[0])*1.*(x[0] <= 5+{BL})*({HW} <= x[1])*(x[1] <= 1)',
            dim_domain=2)}

    boundary_info = {
        'dune': FunctionBasedBoundaryInfo(grid['dune'], default_boundary_type=DirichletBoundary()),
        'pymor': GenericBoundaryInfo.from_indicators(
            grid['pymor'],
            {'dirichlet': lambda X: float_cmp(outflow['pymor'].evaluate(X), 0),
             'neumann': lambda X: float_cmp(outflow['pymor'].evaluate(X), 1)})}
    boundary_info['dune'].register_new_function(outflow['dune'], NeumannBoundary())

    def restricted_advection(x):
        result = np.zeros(x.shape)
        result[..., 0] = channel['pymor'].evaluate(x)
        return result

    problem = {
        'dune': InstationaryDuneProblem(
            stationary_part=StationaryDuneProblem(
                grid['dune'],
                boundary_info['dune'],
                rhs=ConstantFunction(0, dim_domain=2),
                diffusion=channel['dune'] + washcoat['dune']*spe10_perm['dune'],
                advection=LincombDuneGridFunction(
                    functions=[GF(grid['dune'], [1., 0.])*channel['dune'],],
                    coefficients=[ProjectionParameterFunctional('Pe'),]),
                reaction=LincombDuneGridFunction(
                    functions=[washcoat['dune'],],
                    coefficients=[ProjectionParameterFunctional('Da'),]),
                dirichlet_data=inflow['dune'],
                neumann_data=ConstantFunction(0., dim_domain=2),
                outputs=(('l2_boundary', outflow['dune']*GF(grid['dune'], 1./(1 - HW))),),
                name='Spe10ChannelProblem',
                data_approximation_order=0),
            initial_data=inflow['dune'] if not channel_initially_filled else channel['dune'],
            T=T_end[regime]),
        'pymor': InstationaryProblem(
            stationary_part=StationaryProblem(
                domain=RectDomain(domain),
                rhs=ConstantFunction(0, dim_domain=2),
                diffusion=channel['pymor'] + washcoat['pymor']*spe10_perm['pymor'],
                advection=GenericFunction(mapping=restricted_advection,
                                          dim_domain=2, shape_range=(2,))*ProjectionParameterFunctional('Pe'),
                reaction=washcoat['pymor']*ProjectionParameterFunctional('Da'),
                dirichlet_data=inflow['pymor'],
                neumann_data=ConstantFunction(0, dim_domain=2),
                outputs=(('l2_boundary', outflow['pymor']*(1./(1 - HW))),),
                name='Spe10ChannelProblem'),
            initial_data=inflow['pymor'] if not channel_initially_filled else channel['pymor'],
            T=T_end[regime]),
    }

    parameter_space = parameter_ranges[regime]
    mu_bar = parameter_space.parameters.parse({kk: vv for kk, (vv, _) in parameter_space.ranges.items()})

    return grid, num_grid_elements, boundary_info, problem, parameter_space, mu_bar


def discretize(grid, num_grid_elements, boundary_info, problem, mu_bar, nt=127):
    """Yields fom, fom_data, coercivity_estimate."""
    theta_diffusion = ConstantParameterFunctional(1)
    theta_reaction = MinThetaParameterFunctional((ProjectionParameterFunctional('Da'),), mu_bar)
    coercivity_estimate = GenericParameterFunctional(
        mapping=lambda mu: min(theta_diffusion.evaluate(mu), theta_reaction.evaluate(mu)),
        parameters=problem['pymor'].parameters)

    fom, fom_data = discretize_instationary_cg_dune(
        problem['dune'], nt=nt, mu_energy_product=mu_bar, order=1)
    fom = to_numpy(fom, num_grid_elements=num_grid_elements)
    fom = fom.with_(operator=simplify(fom.operator), rhs=simplify(fom.rhs))

    return fom, fom_data, coercivity_estimate
