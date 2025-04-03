# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import threading
from functools import partial
from itertools import chain
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib.colors import TABLEAU_COLORS as COLORS

#from matplotlib.colors import CSS4_COLORS as COLORS
from pymor.core.config import config

config.require('IPYWIDGETS')

from IPython import display
from ipywidgets import HTML as HTMLWIDGET
from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    FloatLogSlider,
    FloatSlider,
    HBox,
    Label,
    Layout,
    Output,
    Stack,
    Text,
    VBox,
    jsdlink,
)

from pymor.core.base import BasicObject
from pymor.models.basic import StationaryModel
from pymor.parameters.base import Mu, Parameters, ParameterSpace


class ParameterSelector(BasicObject):
    """Parameter selector."""

    def __init__(self, space, time_dependent):
        assert isinstance(space, ParameterSpace)
        self.space = space
        self._handlers = []

        class ParameterWidget:
            def __init__(self, p):
                dim = space.parameters[p]
                low, high = space.ranges[p]
                self._sliders = sliders = []
                self._texts = texts = []
                self._checkboxes = checkboxes = []
                stacks = []
                hboxes = []
                for i in range(dim):
                    sliders.append(FloatSlider((high+low)/2, min=low, max=high,
                                               description=f'{i}:'))

                for i in range(dim):
                    texts.append(Text(f'{(high+low)/2:.2f}', description=f'{i}:'))

                    def text_changed(change):
                        try:
                            Parameters(p=1).parse(change['new'])
                            change['owner'].style.background = '#FFFFFF'
                            return
                        except ValueError:
                            pass
                        change['owner'].style.background = '#FFCCCC'

                    texts[i].observe(text_changed, 'value')

                    stacks.append(Stack(children=[sliders[i], texts[i]], selected_index=0, layout=Layout(flex='1')))
                    checkboxes.append(Checkbox(value=False, description='time dep.',
                                               indent=False, layout=Layout(flex='0')))

                    def check_box_clicked(change, idx):
                        stacks[idx].selected_index = int(change['new'])

                    checkboxes[i].observe(lambda change, idx=i: check_box_clicked(change, idx), 'value')

                    hboxes.append(HBox([stacks[i], checkboxes[i]]))
                widgets = VBox(hboxes if time_dependent else sliders)
                self.widget = Accordion(titles=[f'Parameter: {p}'], children=[widgets], selected_index=0)
                self._old_values = [s.value for s in sliders]
                self.valid = True
                self._handlers = []

                for obj in chain(sliders, texts, checkboxes):
                    obj.observe(self._values_changed, 'value')

            def _call_handlers(self):
                for handler in self._handlers:
                    handler()

            def _values_changed(self, change):
                was_valid = self.valid
                new_values = [t.value if c.value else s.value
                              for s, t, c in zip(self._sliders, self._texts, self._checkboxes)]
                # do nothing if new values are invalid
                try:
                    Parameters(p=len(self._sliders)).parse(new_values)
                except ValueError:
                    self.valid = False
                    if was_valid:
                        self._call_handlers()
                    return

                self.valid = True
                if new_values != self._old_values:
                    self._old_values = new_values
                    self._call_handlers()

            @property
            def values(self):
                return self._old_values

            def on_change(self, handler):
                self._handlers.append(handler)

        self._widgets = _widgets = {p: ParameterWidget(p) for p in space.parameters}
        for w in _widgets.values():
            w.on_change(self._update_mu)
        self._auto_update = Checkbox(value=False, indent=False, description='auto update',
                                     layout=Layout(flex='0'))
        self._update_button = Button(description='Update', disabled=False)
        self._update_button.on_click(self._on_update)
        jsdlink((self._auto_update, 'value'), (self._update_button, 'disabled'))
        controls = HBox([self._auto_update, self._update_button],
                        layout=Layout(border='solid 1px lightgray',
                                      margin='2px',
                                      padding='2px',
                                      justify_content='space-around'))
        self.widget = VBox([w.widget for w in _widgets.values()] + [controls])
        self._update_mu()
        self.last_mu = self.mu

    def display(self):
        return display.display(self.widget)

    def on_change(self, handler):
        self._handlers.append(handler)

    def _call_handlers(self):
        self._update_button.disabled = True
        for handler in self._handlers:
            handler(self.mu)
        self.last_mu = self.mu

    def set_param(self, mu):
        self.mu = mu
        for w, param in zip(self._widgets.values(), mu):
            for slider, p in zip(w._sliders, mu[param]):
                slider.value = p

    def _update_mu(self):
        if any(not w.valid for w in self._widgets.values()):
            self._update_button.disabled = True
            return
        self.mu = self.space.parameters.parse({p: w.values for p, w in self._widgets.items()})
        if self._auto_update.value:
            self._call_handlers()
        else:
            self._update_button.disabled = False

    def _on_update(self, b):
        self._call_handlers()


def interact(model, parameter_space, show_solution=True, visualizer=None, transform=None):
    """Interactively explore |Model| in jupyter environment.

    This method dynamically creates a set of `ipywidgets` to interactively visualize
    a model's solution and output.

    Parameters
    ----------
    model
        The |Model| to interact with.
    parameter_space
        |ParameterSpace| within which the |Parameters| of the model can be chosen.
    show_solution
        If `True`, show the model's solution for the given parameters.
    visualizer
        A method of the form `visualize(U, return_widget=True)` which is called to obtain
        an `ipywidget` that renders the solution. If `None`, `model.visualize` is used.
    transform
        A method `transform(U, mu)` returning the data that is passed to the `visualizer`.
        If `None` the solution `U` is passed directly.

    Returns
    -------
    The created widgets as a single `ipywidget`.
    """
    assert model.parameters == parameter_space.parameters
    if model.dim_input > 0:
        params = Parameters(model.parameters, input=model.dim_input)
        parameter_space = ParameterSpace(params, dict(parameter_space.ranges, input=[-1,1]))
    left_pane = []
    parameter_selector = ParameterSelector(parameter_space, time_dependent=not isinstance(model, StationaryModel))
    left_pane.append(parameter_selector.widget)

    has_output = model.dim_output > 0
    tic = perf_counter()
    mu = parameter_selector.mu
    input = parameter_selector.mu.get('input', None)
    mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
            for k in mu if k != 'input'})
    data = model.compute(solution=show_solution, output=has_output, input=input, mu=mu)
    sim_time = perf_counter() - tic

    if has_output:
        output = data['output']
        if len(output) > 1:
            from IPython import get_ipython
            from matplotlib import pyplot as plt
            get_ipython().run_line_magic('matplotlib', 'widget')
            plt.ioff()
            fig, ax = plt.subplots(1,1)
            fig.canvas.header_visible = False
            fig.canvas.layout.flex = '1 0 320px'
            fig.set_figwidth(320 / 100)
            fig.set_figheight(200 / 100)
            output_lines = ax.plot(output)
            fig.legend([str(i) for i in range(model.dim_output)])
            output_widget = fig.canvas
        else:
            labels = [Text(str(o), description=f'{i}:', disabled=True) for i, o in enumerate(output.ravel())]
            output_widget = VBox(labels)
        left_pane.append(Accordion(titles=['output'], children=[output_widget], selected_index=0))

    sim_time_widget = Label(f'{sim_time}s')
    left_pane.append(HBox([Label('simulation time:'), sim_time_widget]))

    left_pane = VBox(left_pane)

    if show_solution:
        U = data['solution']
        if transform:
            U = transform(U, mu)
        visualizer = (visualizer or model.visualize)(U, return_widget=True)
        visualizer.layout.flex = '0.6 0 auto'
        left_pane.layout.flex = '0.4 1 auto'
        widget = HBox([visualizer, left_pane])
        widget.layout.grid_gap = '5%'
    else:
        widget = left_pane

    def do_update(mu):
        if 'input' in mu:
            input = mu.get_time_dependent_value('input') if mu.is_time_dependent('input') else mu['input']
        else:
            input = None
        mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
                for k in mu if k != 'input'})
        tic = perf_counter()
        data = model.compute(solution=show_solution, output=has_output, input=input, mu=mu)
        sim_time = perf_counter() - tic
        if show_solution:
            U = data['solution']
            if transform:
                U = transform(U, mu)
            visualizer.set(U)
        if has_output:
            output = data['output']
            if len(output) > 1:
                for l, o in zip(output_lines, output.T):
                    l.set_ydata(o)
                low, high = ax.get_ylim()
                ax.set_ylim(min(low, np.min(output)), max(high, np.max(output)))
                output_widget.draw_idle()
            else:
                for l, o in zip(output_widget.children, output.ravel()):
                    l.value = str(o)
        sim_time_widget.value = f'{sim_time}s'

    parameter_selector.on_change(do_update)

    return widget


def interact_model_hierarchy(model_hierarchy, parameter_space, model_names, output_function=None,
                             objective_function=None, optimal_parameter=None, optimization_bg_image=None,
                             optimization_bg_image_limits=None, show_solution=True, visualizer=None,
                             optimization_method='Nelder-Mead', optimization_options={}):
    """Interactively explore |Model| in jupyter environment.

    This method dynamically creates a set of `ipywidgets` to interactively visualize
    a model's solution and output.

    Parameters
    ----------
    model
        The |Model| to interact with.
    parameter_space
        |ParameterSpace| within which the |Parameters| of the model can be chosen.
    show_solution
        If `True`, show the model's solution for the given parameters.
    visualizer
        A method of the form `visualize(U, return_widget=True)` which is called to obtain
        an `ipywidget` that renders the solution. If `None`, `model.visualize` is used.
    transform
        A method `transform(U, mu)` returning the data that is passed to the `visualizer`.
        If `None` the solution `U` is passed directly.

    Returns
    -------
    The created widgets as a single `ipywidget`.
    """
    assert model_hierarchy.parameters == parameter_space.parameters
    if model_hierarchy.dim_input > 0:
        params = Parameters(model_hierarchy.parameters, input=model_hierarchy.dim_input)
        parameter_space = ParameterSpace(params, dict(parameter_space.ranges, input=[-1,1]))
    right_pane = []
    left_pane = []

    # Tolerance
    high = 0
    low = -6
    tolerance_slider = FloatLogSlider(value=10**((low+high)/2), min=low, max=high, description='Tolerance:')
    tolerance_update_button = Button(description='Update', disabled=False)
    global num_tols
    num_tols = -1
    global tols
    tols = []

    def do_tolerance_update(_):
        tol = tolerance_slider.value
        global num_tols
        num_tols += 1
        global tols
        tols.append(tol)
        model_hierarchy.set_tolerance(tol)
    do_tolerance_update(None)

    tolerance_update_button.on_click(do_tolerance_update)
    left_pane.append(Accordion(titles=['Tolerance'],
                               children=[HBox([tolerance_slider, tolerance_update_button])],
                               selected_index=0))

    # Parameter selector
    parameter_selector = ParameterSelector(parameter_space, time_dependent=not isinstance(model_hierarchy,
                                                                                          StationaryModel))
    left_pane.append(Accordion(titles=['Manual parameter selection'],
                               children=[parameter_selector.widget],
                               selected_index=0))

    assert len(model_names) == model_hierarchy.num_models

    has_output = model_hierarchy.dim_output > 0
    mu = parameter_selector.mu
    input = parameter_selector.mu.get('input', None)
    mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
            for k in mu if k != 'input'})
    data = model_hierarchy.compute(solution=show_solution, output=has_output, input=input, mu=mu)
    mod_num = data['model_number']

    colors = list(COLORS)

    global global_counter
    global_counter = 1

    from IPython import get_ipython
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'ipympl')
    plt.ioff()

    from matplotlib import markers
    marker_styles = list(markers.MarkerStyle.markers.keys())

    outputs = []
    inputs_to_outputs = []

    # Solution
    if show_solution:
        U = data['solution']
        mod_num = data['model_number']
        U = model_hierarchy.reconstruct(U, mod_num)
        visualizer = (visualizer or model_hierarchy.visualize)#
        visualizer_widget = visualizer.visualize(U[-1], return_widget=True)
        visualizer_widget.layout.flex = '0.6 0 auto'
        right_pane.append(Accordion(titles=['Solution'], children=[visualizer_widget], selected_index=0))

    # Output
    output_scalar = False
    if has_output:
        output = data['output'].ravel()
        if output.shape == (1,) or output_function:
            output_scalar = True
            if output_function:
                output = output_function(output)

            fig_output, ax_output = plt.subplots(1, 1)
            fig_output.canvas.header_visible = False
            #fig_output.canvas.layout.width = '50%'
            fig_output.canvas.layout.flex = '1 0 320px'
            fig_output.set_figwidth(320 / 100)
            fig_output.set_figheight(200 / 100)
            for k, name in enumerate(model_names):
                ax_output.scatter([], [], c=colors[k], marker=marker_styles[0], label=name)
            ax_output.scatter([global_counter], [output], c=colors[mod_num], marker=marker_styles[1])

            ax_output.plot([0], [0], c=colors[-1], label='Estimated mean')
            ax_output.plot([0], [0], c=colors[-2], label='Estimated variance')
            line_mean_estimates = ax_output.plot([0], [0], c=colors[-1], marker=marker_styles[1])[0]
            line_variance_estimates = ax_output.plot([0], [0], c=colors[-2], marker=marker_styles[1])[0]
            fig_output.legend()
            output_widget = fig_output.canvas
            right_pane.append(Accordion(titles=['Outputs'], children=[output_widget], selected_index=0))

    # Statistics
    statistics_out = Output()

    def do_update_statistics(s_out):
        temp = np.array(model_hierarchy.num_successful_calls)
        temp[temp == 0] = 1.
        data_dict = {'Number of evaluations': model_hierarchy.num_successful_calls,
                     'Average runtime': np.array(model_hierarchy.runtimes) / temp,
                     'Training time': model_hierarchy.training_times}
        statistics_table = pd.DataFrame(data=data_dict, index=model_names)
        s_out.outputs = ()
        s_out.append_display_data(statistics_table)

    do_update_statistics(statistics_out)
    right_pane.append(Accordion(titles=['Evaluation statistics'], children=[statistics_out], selected_index=0))

    # Model information
    model_information_out = Output()

    def do_update_model_information(m_out):
        model_information_table = VBox([HTMLWIDGET(value=f'<p><b>{model_name}:</b> {model}</p>',
                                                   layout=Layout(width='90%'))
                                        for model_name, model in zip(model_names, model_hierarchy.models)])
        m_out.outputs = ()
        m_out.append_display_data(model_information_table)

    do_update_model_information(model_information_out)
    right_pane.append(Accordion(titles=['Information on models'], children=[model_information_out], selected_index=0))

    # Scenarios
    scenarios = []
    scenarios_titles = []
    scenarios_numbers = {}
    ## Monte Carlo
    if has_output and output_scalar:
        button_start_monte_carlo = Button(description='Start', disabled=False)
        button_stop_monte_carlo = Button(description='Stop', disabled=True)
        label_current_number_samples = Label('Number of samples: ')
        label_current_estimated_mean = Label('Current estimated mean: ')
        label_current_estimated_variance = Label('Current estimated variance: ')
        current_number_samples = Label('0')
        current_estimated_mean = Label('-')
        current_estimated_variance = Label('-')
        mean_estimates = []
        variance_estimates = []
        global monte_carlo_running
        monte_carlo_running = False

        scenarios.append(VBox([HBox([button_start_monte_carlo, button_stop_monte_carlo]),
                               HBox([label_current_number_samples, current_number_samples]),
                               HBox([label_current_estimated_mean, current_estimated_mean]),
                               HBox([label_current_estimated_variance, current_estimated_variance])]))
        scenarios_titles.append('Monte Carlo estimation')
        scenarios_numbers['monte_carlo'] = len(scenarios_numbers)
    ## Parameter optimization
    if objective_function:
        button_start_optimization = Button(description='Start', disabled=False)
        button_stop_optimization = Button(description='Stop', disabled=True)
        label_current_objective_function_value = Label('Objective function value: ')
        current_objective_function_value = Label('')
        label_current_optimization_parameter = Label('Current parameter: ')
        current_optimization_parameter = Label('')
        global optimization_running
        optimization_running = False

        fig_objective_functional_value, ax_objective_functional_value = plt.subplots(1, 1)
        fig_objective_functional_value.suptitle('Objective function value')
        fig_objective_functional_value.canvas.header_visible = False
        #fig_objective_functional_value.canvas.layout.width = '50%'
        fig_objective_functional_value.canvas.layout.flex = '1 0 320px'
        fig_objective_functional_value.set_figwidth(320 / 100)
        fig_objective_functional_value.set_figheight(200 / 100)
        for k, name in enumerate(model_names):
            ax_objective_functional_value.scatter([], [], c=colors[k], marker=marker_styles[0], label=name)
        fig_objective_functional_value.legend()
        ax_objective_functional_value.set_yscale('log')
        ax_objective_functional_value.set_xlabel('Optimization step')
        objective_function_value_widget = fig_objective_functional_value.canvas
        if model_hierarchy.parameters.dim == 2:
            fig_current_optimization_parameter, ax_current_optimization_parameter = plt.subplots(1, 1)
            fig_current_optimization_parameter.suptitle('Trajectory in parameter space')
            fig_current_optimization_parameter.canvas.header_visible = False
            #fig_current_optimization_parameter.canvas.layout.width = '50%'
            fig_current_optimization_parameter.canvas.layout.flex = '1 0 320px'
            fig_current_optimization_parameter.set_figwidth(320 / 100)
            fig_current_optimization_parameter.set_figheight(200 / 100)

            parameter_bounds = []
            for kk in parameter_space.ranges:
                for jj in range(parameter_space.parameters[kk]):
                    parameter_bounds.append((parameter_space.ranges[kk][0], parameter_space.ranges[kk][1]))
            assert len(parameter_bounds) == 2

            if optimization_bg_image:
                bg_img = plt.imread(optimization_bg_image)
                ax_current_optimization_parameter.imshow(bg_img, extent=[parameter_bounds[0][0], parameter_bounds[0][1],
                                                                         parameter_bounds[1][0], parameter_bounds[1][1]])
                if optimization_bg_image_limits is not None:
                    colormap = plt.cm.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=colormap)
                    sm.set_clim(vmin=optimization_bg_image_limits[0], vmax=optimization_bg_image_limits[1])
                    fig_current_optimization_parameter.colorbar(sm, ax=ax_current_optimization_parameter)

            for k, name in enumerate(model_names):
                ax_current_optimization_parameter.scatter([], [], c=colors[k], marker=marker_styles[0], label=name)
            if optimal_parameter:
                ax_current_optimization_parameter.scatter([optimal_parameter.to_numpy()[0]],
                                                          [optimal_parameter.to_numpy()[1]],
                                                          c="red", marker="x", label="Optimum")
            fig_current_optimization_parameter.legend()
            ax_current_optimization_parameter.set_xlim(parameter_bounds[0][0], parameter_bounds[0][1])
            ax_current_optimization_parameter.set_ylim(parameter_bounds[1][0], parameter_bounds[1][1])
            current_optimization_parameter_widget = fig_current_optimization_parameter.canvas

        scenarios.append(VBox([HBox([button_start_optimization, button_stop_optimization]),
                               HBox([label_current_objective_function_value, current_objective_function_value]),
                               HBox([label_current_optimization_parameter, current_optimization_parameter]),
                               objective_function_value_widget,
                               current_optimization_parameter_widget]))
        scenarios_titles.append('Parameter optimization')
        scenarios_numbers['parameter_optimization'] = len(scenarios_numbers)

    scenarios_accordion = Accordion(titles=['Application scenarios'],
                                    children=[Accordion(titles=scenarios_titles,
                                                        children=scenarios, selected_index=0)],
                                    selected_index=0)

    if has_output and output_scalar:
        def run_monte_carlo(s_out, m_out):
            global monte_carlo_running
            while monte_carlo_running:
                mu = parameter_space.sample_randomly()
                do_parameter_update(mu, s_out, m_out)

        def do_start_monte_carlo(_):
            global monte_carlo_running
            if not monte_carlo_running:
                monte_carlo_running = True
            button_start_monte_carlo.disabled = True
            button_stop_monte_carlo.disabled = False
            if objective_function:
                button_start_optimization.disabled = True
                button_stop_optimization.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['monte_carlo'],
                                                      'Running: Monte Carlo estimation')

            from pymor.tools.random import spawn_rng
            thread_monte_carlo = threading.Thread(target=spawn_rng(run_monte_carlo),
                                                  args=(statistics_out, model_information_out))
            if not thread_monte_carlo.is_alive():
                thread_monte_carlo.start()

        def do_stop_monte_carlo(_):
            global monte_carlo_running
            if monte_carlo_running:
                monte_carlo_running = False
            button_start_monte_carlo.disabled = False
            button_stop_monte_carlo.disabled = True
            if objective_function:
                button_start_optimization.disabled = False
                button_stop_optimization.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['monte_carlo'], 'Monte Carlo estimation')

        button_start_monte_carlo.on_click(do_start_monte_carlo)
        button_stop_monte_carlo.on_click(do_stop_monte_carlo)

    if objective_function:
        class OptimizationInterruptedError(Exception):
            pass

        collected_optimization_data = []

        # TODO: Allow to set initial guess via sliders! Activate start button afterwards!
        global initial_guess
        initial_guess = mu.to_numpy()

        global optimization_interrupted
        optimization_interrupted = False

        parameter_bounds = []
        for kk in parameter_space.ranges:
            for jj in range(parameter_space.parameters[kk]):
                parameter_bounds.append((parameter_space.ranges[kk][0], parameter_space.ranges[kk][1]))
        parameter_bounds = np.array(tuple(np.array(b) for b in parameter_bounds))

        def run_optimization():
            from scipy.optimize import minimize as scipy_optimize
            global initial_guess
            while True:
                try:
                    optimization_results = scipy_optimize(partial(do_parameter_update, s_out=statistics_out,
                                                                  m_out=model_information_out),
                                                          x0=initial_guess, method=optimization_method,
                                                          bounds=parameter_bounds, options=optimization_options)
                    print(f"Optimization result: {optimization_results}")
                    do_stop_optimization(None)
                    break
                except OptimizationInterruptedError:
                    last_mu = collected_optimization_data[-1]['point']
                    initial_guess = last_mu.to_numpy()
                    global optimization_running
                    optimization_running = False

        def do_start_optimization(_):
            global optimization_running
            if not optimization_running:
                optimization_running = True
                global optimization_interrupted
                optimization_interrupted = False
            button_start_optimization.disabled = True
            button_stop_optimization.disabled = False
            if has_output and output_scalar:
                button_start_monte_carlo.disabled = True
                button_stop_monte_carlo.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['parameter_optimization'],
                                                      'Running: Parameter optimization')

            from pymor.tools.random import spawn_rng
            thread_optimization = threading.Thread(target=spawn_rng(run_optimization))
            if not thread_optimization.is_alive():
                thread_optimization.start()

        def do_stop_optimization(_):
            global optimization_running
            if optimization_running:
                global optimization_interrupted
                optimization_interrupted = True
            button_start_optimization.disabled = False
            button_stop_optimization.disabled = True
            if has_output and output_scalar:
                button_start_monte_carlo.disabled = False
                button_stop_monte_carlo.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['parameter_optimization'],
                                                      'Parameter optimization')

        button_start_optimization.on_click(do_start_optimization)
        button_stop_optimization.on_click(do_stop_optimization)

    left_pane.append(scenarios_accordion)

    # Error estimates
    fig_error_estimates, ax_error_estimates = plt.subplots(1, 1)
    fig_error_estimates.canvas.header_visible = False
    #fig_error_estimates.canvas.layout.width = '50%'
    fig_error_estimates.canvas.layout.flex = '1 0 320px'
    fig_error_estimates.set_figwidth(320 / 100)
    fig_error_estimates.set_figheight(200 / 100)
    for k, name in enumerate(model_names[:-1]):
        ax_error_estimates.scatter([], [], c=colors[k], label=f'{name}')
    for k, (est_err, name) in enumerate(zip(data['error_estimates'], model_names[:-1])):
        if est_err is not None:
            ax_error_estimates.scatter([global_counter], [est_err], c=colors[k])
    inputs_to_tolerances = [global_counter]
    tolerances = [model_hierarchy.get_tolerance()]
    line_tolerances = ax_error_estimates.plot(inputs_to_tolerances, tolerances, c=colors[-1], label='Tolerance')[0]
    fig_error_estimates.legend()
    ax_error_estimates.set_yscale('log')
    error_estimates_widget = fig_error_estimates.canvas
    error_estimates_accordion = Accordion(titles=['Error estimates'], children=[error_estimates_widget],
                                          selected_index=0)
    left_pane.append(error_estimates_accordion)

    # Timings
    fig_timings, ax_timings = plt.subplots(1, 1)
    fig_timings.canvas.header_visible = False
    #fig_timings.canvas.layout.width = '50%'
    fig_timings.canvas.layout.flex = '1 0 320px'
    fig_timings.set_figwidth(320 / 100)
    fig_timings.set_figheight(200 / 100)
    for k, (runtime, name) in enumerate(zip(data['runtimes'], model_names)):
        if k > 0:
            ax_timings.bar(global_counter, runtime, bottom=np.sum(data['runtimes'][:k]), color=colors[k],
                           edgecolor='black', label=f'{name} evaluation')
        else:
            ax_timings.bar(global_counter, runtime, color=colors[k], edgecolor='black', label=f'{name} evaluation')
    for k, name in enumerate(model_names[:-1]):
        ax_timings.bar(global_counter, 0., color=colors[k], edgecolor='black', hatch='//', label=f'{name} training')
    for k, training_time in enumerate(data['training_times']):
        ax_timings.bar(global_counter, training_time, bottom=np.sum(data['runtimes']), color=colors[k],
                       edgecolor='black', hatch='//')
    ax_timings.set_yscale('log')
    fig_timings.legend()
    timings_widget = fig_timings.canvas
    left_pane.append(Accordion(titles=['Timings'], children=[timings_widget], selected_index=0))

    right_pane = VBox(right_pane)
    right_pane.layout.width = '50%'
    left_pane = VBox(left_pane)
    left_pane.layout.width = '50%'
    widget = HBox([left_pane, right_pane])
    widget.layout.grid_gap = '2%'

    def do_parameter_update(mu, s_out, m_out):
        global global_counter
        global_counter = global_counter + 1

        mu = model_hierarchy.parameters.parse(mu)
        if 'input' in mu:
            input = mu.get_time_dependent_value('input') if mu.is_time_dependent('input') else mu['input']
        else:
            input = None
        mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
                for k in mu if k != 'input'})

        parameter_selector.set_param(mu)

        data = model_hierarchy.compute(solution=show_solution, output=has_output, input=input, mu=mu)
        mod_num = data['model_number']
        if show_solution:
            U = data['solution']
            visualizer.set(model_hierarchy.reconstruct(U[-1], mod_num))
            visualizer_widget.draw()
        if has_output and output_scalar:
            output = data['output'].ravel()
            if output_function:
                output = output_function(output)
            outputs.append(output)
            ax_output.scatter([global_counter], [output], c=colors[mod_num], marker=marker_styles[1])
            low, high = ax_output.get_ylim()
            ax_output.set_ylim(min(low, output), max(high, output))
            output_widget.draw()

            global monte_carlo_running
            if monte_carlo_running:
                inputs_to_outputs.append(global_counter)
                current_number_samples.value = str(len(outputs))
                mean_estimate = np.mean(np.array(outputs), axis=0)
                mean_estimates.append(mean_estimate)
                current_estimated_mean.value = str(np.mean(np.array(outputs), axis=0))
                variance_estimate = np.var(np.array(outputs), axis=0)
                variance_estimates.append(variance_estimate)
                current_estimated_variance.value = str(np.var(np.array(outputs), axis=0))
                line_mean_estimates.set_data(np.array(inputs_to_outputs), np.array(mean_estimates))
                line_variance_estimates.set_data(np.array(inputs_to_outputs), np.array(mean_estimates))

        for k, (est_err, name) in enumerate(zip(data['error_estimates'], model_names)):
            if est_err is not None:
                ax_error_estimates.scatter([global_counter], [est_err], c=colors[k])
        low, high = ax_error_estimates.get_ylim()
        arr_err_ests = np.array(data['error_estimates'])
        ax_error_estimates.set_ylim(min(low, np.min(arr_err_ests[arr_err_ests != np.array(None)]) * 0.9),
                                    max(high, np.max(arr_err_ests[arr_err_ests != np.array(None)]) * 1.1))
        error_estimates_widget.draw()

        inputs_to_tolerances.append(global_counter)
        tolerances.append(model_hierarchy.get_tolerance())
        line_tolerances.set_data(np.array(inputs_to_tolerances), np.array(tolerances))

        for k, runtime in enumerate(data['runtimes']):
            if k > 0:
                ax_timings.bar(global_counter, runtime, bottom=np.sum(data['runtimes'][:k]), color=colors[k],
                               edgecolor='black')
            else:
                ax_timings.bar(global_counter, runtime, color=colors[k], edgecolor='black')
        for k, training_time in enumerate(data['training_times']):
            ax_timings.bar(global_counter, training_time, bottom=np.sum(data['runtimes']), color=colors[k],
                           edgecolor='black', hatch='//')
        _, high = ax_timings.get_ylim()
        ax_timings.set_ylim(0., max(high, (np.sum(data['runtimes']) + np.sum(data['training_times'])) * 1.1))
        timings_widget.draw()

        do_update_statistics(s_out)
        do_update_model_information(m_out)

        if objective_function:
            global optimization_running
            if optimization_running:
                global optimization_interrupted
                if optimization_interrupted:
                    raise OptimizationInterruptedError
                else:
                    # TODO: Update plots of optimization trajectory
                    # in parameter space and objective function values!
                    quantity_of_interest = objective_function(model_hierarchy, mu)
                    current_objective_function_value.value = str(quantity_of_interest)
                    current_optimization_parameter.value = str(mu.to_numpy())

                    if model_hierarchy.parameters.dim == 2:
                        ax_current_optimization_parameter.scatter([mu.to_numpy()[0]], [mu.to_numpy()[1]], c=colors[num_tols], label=f"Tol: {tols[-1]}")
                        current_optimization_parameter_widget.draw()

                    # TODO: Get model used for computation or tolerance values... and use corresponding color to distinguish them!
                    # TODO: Get optimization iteration and not global_counter!
                    ax_objective_functional_value.scatter([global_counter], [quantity_of_interest], c=colors[0], marker=marker_styles[0])
                    low, high = ax_objective_functional_value.get_ylim()
                    ax_objective_functional_value.set_ylim(min(low, quantity_of_interest) * 0.9,
                                                           max(high, quantity_of_interest) * 1.1)
                    objective_function_value_widget.draw()

                    collected_optimization_data.append({'point': mu, 'val': quantity_of_interest})
                    return quantity_of_interest

    def do_manual_parameter_update(mu, s_out, m_out):
        global initial_guess
        initial_guess = mu.to_numpy()
        return do_parameter_update(mu, s_out, m_out)

    parameter_selector.on_change(partial(do_manual_parameter_update, s_out=statistics_out, m_out=model_information_out))

    return widget


    # TODO: Mention current status (solving, training, estimating, etc.) somewhere
    # TODO: Implement parameter optimization example
    # TODO: Change parameter sliders when using Monte Carlo
    # TODO: If 2d parameter space: Select parameter by clicking on parameter set
    # + visualize selected parameters + non-uniform density for parameter selection in Monte Carlo
