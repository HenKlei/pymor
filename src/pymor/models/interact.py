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


def interact_model_hierarchy(model_hierarchy, parameter_space, model_names, show_solution=True, visualizer=None):
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

    def do_tolerance_update(_):
        tol = tolerance_slider.value
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

    outputs = []

    # Solution
    if show_solution:
        U = data['solution']
        mod_num = data['model_number']
        U = model_hierarchy.reconstruct(U, mod_num)
        visualizer = (visualizer or model_hierarchy.visualize)(U, return_widget=True)
        visualizer.layout.flex = '0.6 0 auto'
        right_pane.append(Accordion(titles=['Solution'], children=[visualizer], selected_index=0))

    # Output
    if has_output:
        output = data['output']
        dim_output = model_hierarchy.dim_output
        from matplotlib import markers
        marker_styles = list(markers.MarkerStyle.markers.keys())
        assert len(output) == 1
        outputs.append(output[0])
        plt.ioff()
        fig, ax_output = plt.subplots(1, 1)
        fig.canvas.header_visible = False
        #fig.canvas.layout.width = '50%'
        fig.canvas.layout.flex = '1 0 320px'
        fig.set_figwidth(320 / 100)
        fig.set_figheight(200 / 100)
        for k, name in enumerate(model_names):
            for i in range(dim_output):
                ax_output.scatter([], [], c=colors[k], marker=marker_styles[i % len(marker_styles)],
                                  label=f'{name}{": "+str(i) if dim_output>1 else ""}')
        fig.legend()
        for i, o in enumerate(output[0]):
            ax_output.scatter([global_counter], [o], c=colors[mod_num],
                              marker=marker_styles[i % len(marker_styles)], label=f'{model_names[mod_num]}: {i}')
        output_widget = fig.canvas
        right_pane.append(Accordion(titles=['Output'], children=[output_widget], selected_index=0))

    # Timings
    #right_pane.append()

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
    ## Monte Carlo
    button_start_monte_carlo = Button(description='Start', disabled=False)
    button_stop_monte_carlo = Button(description='Stop', disabled=True)
    label_current_number_samples = Label('Number of samples: ')
    label_current_estimated_mean = Label(f'Current estimated mean{"s" if dim_output>1 else ""}: ')
    label_current_estimated_variance = Label(f'Current estimated variance{"s" if dim_output>1 else ""}: ')
    current_number_samples = Label('0')
    current_estimated_mean = Label('-')
    current_estimated_variance = Label('-')
    global monte_carlo_running
    monte_carlo_running = False

    scenarios.append(VBox([HBox([button_start_monte_carlo, button_stop_monte_carlo]),
                           HBox([label_current_number_samples, current_number_samples]),
                           HBox([label_current_estimated_mean, current_estimated_mean]),
                           HBox([label_current_estimated_variance, current_estimated_variance])]))
    ## Parameter optimization
    scenarios.append(VBox())

    scenarios_accordion = Accordion(titles=['Application scenarios'],
                                    children=[Accordion(titles=['Monte Carlo estimation', 'Parameter optimization'],
                                                        children=scenarios, selected_index=0)],
                                    selected_index=0)

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
        scenarios_accordion.children[0].set_title(0, 'Running: Monte Carlo estimation')

        thread_monte_carlo = threading.Thread(target=run_monte_carlo, args=(statistics_out, model_information_out))
        if not thread_monte_carlo.is_alive():
            thread_monte_carlo.start()

    def do_stop_monte_carlo(_):
        global monte_carlo_running
        if monte_carlo_running:
            monte_carlo_running = False
        button_start_monte_carlo.disabled = False
        button_stop_monte_carlo.disabled = True
        scenarios_accordion.children[0].set_title(0, 'Monte Carlo estimation')

    button_start_monte_carlo.on_click(do_start_monte_carlo)
    button_stop_monte_carlo.on_click(do_stop_monte_carlo)

    left_pane.append(scenarios_accordion)

    right_pane = VBox(right_pane)
    right_pane.layout.width = '50%'
    left_pane = VBox(left_pane)
    left_pane.layout.width = '50%'
    widget = HBox([left_pane, right_pane])
    widget.layout.grid_gap = '2%'

    def do_parameter_update(mu, s_out, m_out):
        global global_counter
        global_counter = global_counter + 1

        if 'input' in mu:
            input = mu.get_time_dependent_value('input') if mu.is_time_dependent('input') else mu['input']
        else:
            input = None
        mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
                for k in mu if k != 'input'})
        data = model_hierarchy.compute(solution=show_solution, output=has_output, input=input, mu=mu)
        mod_num = data['model_number']
        if show_solution:
            U = data['solution']
            visualizer.set(model_hierarchy.reconstruct(U, mod_num))
        if has_output:
            output = data['output']
            assert len(output) == 1
            outputs.append(output[0])
            for i, o in enumerate(output[0]):
                ax_output.scatter([global_counter], [o], c=colors[mod_num],
                                  marker=marker_styles[i % len(marker_styles)],
                                  label=f'{model_names[mod_num]}{": "+str(i) if dim_output>1 else ""}')
            low, high = ax_output.get_ylim()
            ax_output.set_ylim(min(low, np.min(output)), max(high, np.max(output)))
            output_widget.draw()
            global monte_carlo_running
            if monte_carlo_running:
                current_number_samples.value = str(len(outputs))
                current_estimated_mean.value = str(np.mean(np.array(outputs), axis=0))
                current_estimated_variance.value = str(np.var(np.array(outputs), axis=0))
                # TODO: Add estimates to output plot!

        do_update_statistics(s_out)
        do_update_model_information(m_out)

    parameter_selector.on_change(partial(do_parameter_update, s_out=statistics_out, m_out=model_information_out))

    return widget


    # TODO: Mention current status (solving, training, estimating, etc.) somewhere
    # TODO: Plot error estimates somewhere
    # TODO: Add estimates of mean and variance to plot
    # TODO: Implement parameter optimization example
    # TODO: Plot tolerance in same plot as error estimates
    # TODO: Change parameter sliders when using Monte Carlo
    # TODO: If 2d parameter space: Select parameter by clicking on parameter set
    # + visualize selected parameters + non-uniform density for parameter selection in Monte Carlo
