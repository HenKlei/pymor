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
    HTML,
    IntSlider,
    Label,
    Layout,
    link,
    Output,
    Play,
    RadioButtons,
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
            fig, ax = plt.subplots(1, 1)
            fig.canvas.header_visible = False
            fig.canvas.layout.flex = '1 0 320px'
            fig.set_figwidth(320 / 100)
            fig.set_figheight(200 / 100)
            output_lines = ax.plot(output)
            fig.legend([str(i) for i in range(model.dim_output)], framealpha=1.)
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
                             objective_function=None, initial_parameter=None, optimal_parameter=None,
                             optimization_bg_image=None, optimization_bg_image_limits=None, show_solution=True,
                             visualizer=None, optimization_method='Nelder-Mead', optimization_options={},
                             random_sampling_function=None, density_function_monte_carlo=None,
                             fig_width=17, fig_height=3, solution_plot_extent=(0, 1, 0, 1), language='en'):
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
    translations = {'Reset': {'de': 'Zurücksetzen'},
                    'Reset all plots (retain model hierarchy)': {'de': 'Alle Diagramme zurücksetzen (Modellhierarchie beibehalten)'},
                    'Update': {'de': 'Aktualisieren'},
                    'Current tolerance': {'de': 'Aktuelle Rechengenauigkeit'},
                    'Tolerance': {'de': 'Rechengenauigkeit'},
                    'Choose tolerance': {'de': 'Wähle Rechengenauigkeit'},
                    'Manual parameter selection': {'de': 'Manuelle Parameterwahl'},
                    'Selection from parameter space': {'de': 'Auswahl im Parameterraum'},
                    'Error estimates': {'de': 'Fehlerschätzungen'},
                    'Evaluation times': {'de': 'Auswertungszeiten'},
                    'Training times': {'de': 'Trainingzeiten'},
                    'Evaluation statistics': {'de': 'Statistiken'},
                    'Solution': {'de': 'Konzentrationsverteilung des Schadstoffs'},
                    'Timings': {'de': 'Laufzeiten'},
                    'Parameter optimization': {'de': 'Parameteroptimierung'},
                    'evaluation': {'de': 'Auswertung'},
                    'training': {'de': 'Training'},
                    'Outputs and estimated statistics': {'de': 'Ausgabewert und geschätzte Statistiken'},
                    'Randomly selected parameters': {'de': 'Zufällig gewählte Parameter'},
                    'Randomly selected parameters and probability density function': {'de': 'Zufällig gewählte Parameter \n und Wahrscheinlichkeitsdichte'},
                    'Estimated mean': {'de': 'Geschätzter Mittelwert'},
                    'Start': {'de': 'Start'},
                    'Stop': {'de': 'Stop'},
                    'Number of samples': {'de': 'Stichprobengröße'},
                    'Current estimated mean': {'de': 'Aktuell geschätzter Mittelwert'},
                    'Current estimated variance': {'de': 'Aktuell geschätzte Varianz'},
                    'Monte Carlo estimation': {'de': 'Abschätzung der Auswirkungen von Materialunsicherheiten'},
                    'Samples': {'de': 'Samples'},
                    'Initialization': {'de': 'Initialisierung'},
                    'Objective function value': {'de': 'Zielfunktionswert'},
                    'Current parameter': {'de': 'Aktueller Parameter'},
                    'Optimization step': {'de': 'Optimierungsschritt'},
                    'Trajectory in parameter space': {'de': 'Trajektorie im Parameterraum'},
                    'Optimum': {'de': 'Optimum'},
                    'Application scenarios': {'de': 'Anwendungsszenario'},
                    'Initial guess': {'de': 'Startpunkt'},
                    'Running': {'de': 'Aktuell'},
                    'Number of evaluations': {'de': 'Anzahl der Auswertungen'},
                    'Average runtime': {'de': 'Durchschnittliche Laufzeit'},
                    'Training time': {'de': 'Trainingszeit'},
                    'Dimension of model': {'de': 'Dimension des Modells'}}

    def translate(phrase):
        return (translations.get(phrase, {language: phrase})).get(language, phrase)


    from IPython import get_ipython
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'ipympl')
    plt.ioff()

    assert model_hierarchy.parameters == parameter_space.parameters
    if model_hierarchy.dim_input > 0:
        params = Parameters(model_hierarchy.parameters, input=model_hierarchy.dim_input)
        parameter_space = ParameterSpace(params, dict(parameter_space.ranges, input=[-1,1]))

    parameter_bounds = []
    for kk in parameter_space.ranges:
        for jj in range(parameter_space.parameters[kk]):
            parameter_bounds.append((parameter_space.ranges[kk][0], parameter_space.ranges[kk][1]))

    list_of_reset_functions = []

    right_pane = []
    left_pane = []

    # Reset button
    reset_button = Button(description=translate('Reset'), disabled=False,
                          layout=Layout(width='250px', height='50px'), button_style='warning')

    def do_reset(_):
        for f in list_of_reset_functions:
            f()

    reset_button.on_click(do_reset)
    right_pane.append(Accordion(titles=[translate('Reset all plots (retain model hierarchy)')],
                                children=[reset_button], selected_index=0))

    # Tolerance
    available_tolerances = [1e-5, 1e-4, 1e-3, 1e-2]
    tolerance_radio_buttons = RadioButtons(options=[(f'{t:.0e}', t) for t in available_tolerances],
                                           value=available_tolerances[len(available_tolerances)//2])
    radio_buttons_style = HTML(
        '<style>.widget-radio {width: auto;}'
        '.widget-radio-box {flex-direction: row !important; float: inline-end; margin-bottom: 20px;}'
        '.widget-radio-box label {margin: 5px !important;}'
        '.widget-radio-box input {margin-left: 5px; transform: scale(1.5); margin-right: 10px;}</style>',
        layout=Layout(display='none'),
    )
    tolerance_update_button = Button(description=translate('Update'),
                                     layout=Layout(width='200px'), disabled=False, button_style='primary')
    tolerance_label = Label(f'{translate("Current tolerance")}: ', layout={'margin': '10px 0px 0px 0px'})
    current_tol_label = Label('', layout={'margin': '10px 0px 0px 5px'})

    global num_tol
    global tols

    def do_tolerance_update(_):
        tol = tolerance_radio_buttons.value
        global num_tol
        num_tol = available_tolerances.index(tol)
        global tols
        tols.append(tol)
        model_hierarchy.set_tolerance(tol)
        current_tol_label.value = f'{tol:.3e}'

    def reset_tols():
        global num_tol
        num_tol = -1
        global tols
        tols = []
        do_tolerance_update(None)

    reset_tols()
    list_of_reset_functions.append(reset_tols)

    tolerance_update_button.on_click(do_tolerance_update)
    left_pane.append(Accordion(titles=[translate('Tolerance')],
                               children=[VBox([HBox([Label(f'{translate("Choose tolerance")}:'),
                                                     tolerance_radio_buttons, radio_buttons_style,
                                                     tolerance_update_button]),
                                               HBox([tolerance_label, current_tol_label])])],
                               selected_index=0))

    # Parameter selector
    parameter_selector = ParameterSelector(parameter_space, time_dependent=not isinstance(model_hierarchy,
                                                                                          StationaryModel))
    parameter_title = translate('Manual parameter selection')
    parameter_widget = parameter_selector.widget

    global manual_selection
    manual_selection = False
    global manual_selection_counter
    manual_selection_counter = 0

    if model_hierarchy.parameters.dim == 2:
        fig_parameter_selection_onclick, ax_parameter_selection_onclick = plt.subplots(1, 1)
        fig_parameter_selection_onclick.suptitle(translate('Selection from parameter space'))
        fig_parameter_selection_onclick.canvas.header_visible = False
        parameter_widget = fig_parameter_selection_onclick.canvas

        def onclick_param(event):
            if not (event.xdata is None or event.ydata is None):
                mu = model_hierarchy.parameters.parse([event.xdata, event.ydata])
                global manual_selection
                manual_selection = True
                global manual_selection_counter
                manual_selection_counter = manual_selection_counter + 1
                mod_num = do_parameter_update(mu, s_out=statistics_out)
                manual_selection = False
                ax_parameter_selection_onclick.scatter([event.xdata], [event.ydata], c=colors[mod_num])
                fig_parameter_selection_onclick.canvas.draw()

        cid = fig_parameter_selection_onclick.canvas.mpl_connect('button_press_event', onclick_param)

        def reset_fig_parameter_selection_onclick():
            ax_parameter_selection_onclick.clear()
            assert len(parameter_bounds) == 2
            fig_parameter_selection_onclick.set_figwidth(fig_width / 2.)
            fig_parameter_selection_onclick.set_figheight(fig_height)
            ax_parameter_selection_onclick.set_xlim(parameter_bounds[0][0], parameter_bounds[0][1])
            ax_parameter_selection_onclick.set_ylim(parameter_bounds[1][0], parameter_bounds[1][1])
            ax_parameter_selection_onclick.set_xlabel(str(list(model_hierarchy.parameters.keys())[0]))
            ax_parameter_selection_onclick.set_ylabel(str(list(model_hierarchy.parameters.keys())[1]))
            fig_parameter_selection_onclick.tight_layout()
            parameter_widget.draw()
            global manual_selection_counter
            manual_selection_counter = 0

        list_of_reset_functions.append(reset_fig_parameter_selection_onclick)

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

    def reset_global_counter():
        global global_counter
        global_counter = 0

    reset_global_counter()
    list_of_reset_functions.append(reset_global_counter)

    from matplotlib import markers
    marker_styles = list(markers.MarkerStyle.markers.keys())

    global outputs
    outputs = []
    global inputs_to_outputs
    inputs_to_outputs = []

    # Solution
    if show_solution:
        U = data['solution']
        mod_num = data['model_number']
        U = model_hierarchy.reconstruct(U, mod_num)
        global current_sol
        current_sol = U
        visualizer = (visualizer or model_hierarchy.visualize)
        visualizer_widget = visualizer.visualize(U[-1], return_widget=True, fig_width=fig_width, fig_height=fig_height,
                                                 extent=solution_plot_extent)

        visualizer_children = [visualizer_widget]
        if len(U) > 1:
            def update_plot(change):
                i = change.new
                global current_sol
                visualizer.set(current_sol[i])
                visualizer_widget.draw()

            fps = 1
            solution_visualizer_player = Play(
                value=len(U)-1,
                min=0,
                max=len(U)-1,
                step=1,
                interval=int(1/round(fps)*1000)  # referesh interval in ms
            )
            solution_time_step_slider = IntSlider(
                value=0,
                min=0,
                max=len(U)-1
            )
            solution_time_step_slider.observe(update_plot, names='value')
            link((solution_visualizer_player, 'value'), (solution_time_step_slider, 'value'))
            visualizer_children = [VBox([visualizer_widget,
                                         HBox([solution_visualizer_player, solution_time_step_slider])])]

        def reset_solution_visualization():
            visualizer.reset()
            visualizer_widget.draw()

        list_of_reset_functions.append(reset_solution_visualization)

        left_pane.append(Accordion(titles=[translate('Solution')], children=visualizer_children, selected_index=0))

    # Error estimates
    fig_error_estimates, ax_error_estimates = plt.subplots(1, 1)
    error_estimates_widget = fig_error_estimates.canvas
    error_estimates_accordion = Accordion(titles=[translate('Error estimates')], children=[error_estimates_widget],
                                          selected_index=0)

    global line_tolerances
    line_tolerances = None
    global inputs_to_tolerances
    inputs_to_tolerances = []
    global tolerances
    tolerances = []

    def reset_fig_error_estimates():
        ax_error_estimates.clear()
        fig_error_estimates.canvas.header_visible = False
        fig_error_estimates.set_figwidth(fig_width)
        fig_error_estimates.set_figheight(fig_height)
        fig_error_estimates.tight_layout()
        fig_error_estimates.legends = []
        for k, name in enumerate(model_names[:-1]):
            ax_error_estimates.scatter([], [], c=colors[k], label=f'{name}')
        for k, (est_err, name) in enumerate(zip(data['error_estimates'], model_names[:-1])):
            if est_err is not None:
                ax_error_estimates.scatter([], [], c=colors[k])
        global inputs_to_tolerances
        inputs_to_tolerances = []
        global tolerances
        tolerances = []
        global line_tolerances
        line_tolerances = ax_error_estimates.plot(inputs_to_tolerances, tolerances, c=colors[-1],
                                                  label=translate('Tolerance'))[0]
        ax_error_estimates.legend(framealpha=1., ncols=3, bbox_to_anchor=(0.5, -0.1), loc='upper center')
        ax_error_estimates.set_yscale('symlog', linthresh=10**(-8))
        fig_error_estimates.subplots_adjust(bottom=0.2)
        error_estimates_widget.draw()

    list_of_reset_functions.append(reset_fig_error_estimates)

    left_pane.append(error_estimates_accordion)

    # Timings
    fig_timings, ax_timings = plt.subplots(1, 2)
    timings_widget = fig_timings.canvas

    def reset_fig_timings():
        ax_timings[0].clear()
        ax_timings[1].clear()
        ax_timings[0].set_title(translate('Evaluation times'))
        ax_timings[1].set_title(translate('Training times'))
        fig_timings.canvas.header_visible = False
        fig_timings.set_figwidth(fig_width)
        fig_timings.set_figheight(fig_height)
        fig_timings.tight_layout()
        fig_timings.legends = []
        for k, name in enumerate(model_names):
            ax_timings[0].bar([0], [0], color=colors[k], edgecolor='black',
                              label=f'{name} {translate("evaluation")}', width=0)
        for k, name in enumerate(model_names[:-1]):
            ax_timings[1].bar([0], [0], color=colors[k], edgecolor='black', hatch='//',
                              label=f'{name} {translate("training")}', width=0)
        ax_timings[0].set_yscale('symlog')
        fig_timings.legend(framealpha=1., ncols=3, bbox_to_anchor=(0.5, 0.02), loc='lower center')
        fig_timings.subplots_adjust(bottom=0.3)
        timings_widget.draw()

    list_of_reset_functions.append(reset_fig_timings)

    left_pane.append(Accordion(titles=[translate('Timings')], children=[timings_widget], selected_index=0))

    # Statistics
    statistics_out = Output()

    def do_update_statistics(s_out):
        temp = np.array(model_hierarchy.num_successful_calls)
        temp[temp == 0] = 1.
        data_dict = {translate('Number of evaluations'): model_hierarchy.num_successful_calls,
                     translate('Average runtime'): np.array(model_hierarchy.runtimes) / temp,
                     translate('Training time'): model_hierarchy.training_times,
                     translate('Dimension of model'): [mod.solution_space.dim if mod is not None else 0
                                                       for mod in model_hierarchy.models]}
        statistics_table = pd.DataFrame(data=data_dict, index=model_names)
        s_out.outputs = ()
        s_out.append_display_data(statistics_table)

    do_update_statistics(statistics_out)
    left_pane.append(Accordion(titles=[translate('Evaluation statistics')],
                               children=[statistics_out], selected_index=0))

    if has_output:
        fig_parameter_selection_output_values, ax_parameter_selection_output_values = plt.subplots(1, 1)
        fig_parameter_selection_output_values.suptitle(translate('Objective function value'))
        fig_parameter_selection_output_values.canvas.header_visible = False
        parameter_selection_output_values_widget = fig_parameter_selection_output_values.canvas

        def reset_fig_parameter_selection_output_values():
            ax_parameter_selection_output_values.clear()
            fig_parameter_selection_output_values.canvas.header_visible = False
            fig_parameter_selection_output_values.set_figwidth(fig_width / 2.)
            fig_parameter_selection_output_values.set_figheight(fig_height)
            fig_parameter_selection_output_values.tight_layout()
            fig_parameter_selection_output_values.legends = []
            for k, name in enumerate(model_names):
                ax_parameter_selection_output_values.scatter([], [], c=colors[k], marker=marker_styles[0], label=name)
            ax_parameter_selection_output_values.legend(framealpha=1.)
            parameter_selection_output_values_widget .draw()

        list_of_reset_functions.append(reset_fig_parameter_selection_output_values)

    # Scenarios
    if has_output:
        scenarios = [HBox([parameter_widget, parameter_selection_output_values_widget])]
    else:
        scenarios = [parameter_widget]
    scenarios_titles = [parameter_title]
    scenarios_numbers = {'manual_parameter_selection': 0}
    global optimization_initialized
    optimization_initialized = False
    ## Parameter optimization
    if objective_function:
        global optimization_iterations
        optimization_iterations = 0

        button_initialization_optimization = Button(description=translate('Initialization'), button_style='primary',
                                                    layout=Layout(width='250px', height='50px'), disabled=False)
        button_start_optimization = Button(description=translate('Start'), button_style='primary',
                                           layout=Layout(width='100px', height='50px'), disabled=True)
        button_stop_optimization = Button(description=translate('Stop'), button_style='primary',
                                          layout=Layout(width='100px', height='50px'), disabled=True)
        label_current_objective_function_value = Label(f'{translate("Objective function value")}: ')
        current_objective_function_value = Label('')
        label_current_optimization_parameter = Label(f'{translate("Current parameter")}: ')
        current_optimization_parameter = Label('')
        global optimization_running
        optimization_running = False

        if model_hierarchy.parameters.dim == 2:
            fig_parameter_optimization, axs = plt.subplots(1, 2)
            ax_objective_functional_value = axs[1]
            ax_current_optimization_parameter = axs[0]
        else:
            fig_parameter_optimization, ax_objective_functional_value = plt.subplots(1, 1)
        parameter_optimization_widget = fig_parameter_optimization.canvas

        if model_hierarchy.parameters.dim == 2:
            if optimization_bg_image:
                if optimization_bg_image_limits is not None:
                    colormap = plt.cm.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=colormap)
                    sm.set_clim(vmin=optimization_bg_image_limits[0], vmax=optimization_bg_image_limits[1])
                    fig_parameter_optimization.colorbar(sm, ax=ax_current_optimization_parameter)

        def reset_fig_parameter_optimization():
            global optimization_initialized
            optimization_initialized = False
            button_start_optimization.disabled = True

            ax_objective_functional_value.clear()
            ax_objective_functional_value.set_title(translate('Objective function value'))
            fig_parameter_optimization.canvas.header_visible = False
            fig_parameter_optimization.set_figwidth(fig_width)
            fig_parameter_optimization.set_figheight(fig_height)
            fig_parameter_optimization.tight_layout()
            fig_parameter_optimization.legends = []
            ax_objective_functional_value.set_xlabel(translate('Optimization step'))

            if model_hierarchy.parameters.dim == 2:
                ax_current_optimization_parameter.clear()
                ax_current_optimization_parameter.set_title(translate('Trajectory in parameter space'))

                assert len(parameter_bounds) == 2

                if optimization_bg_image:
                    bg_img = plt.imread(optimization_bg_image)
                    ax_current_optimization_parameter.imshow(bg_img,
                                                             extent=[parameter_bounds[0][0], parameter_bounds[0][1],
                                                                     parameter_bounds[1][0], parameter_bounds[1][1]])

                if optimal_parameter:
                    ax_current_optimization_parameter.scatter([optimal_parameter.to_numpy()[0]],
                                                              [optimal_parameter.to_numpy()[1]],
                                                              c='red', marker='x',
                                                              label=f'{translate("Optimum")} {optimal_parameter}')
                ax_current_optimization_parameter.legend(framealpha=1., bbox_to_anchor=(0.5, -0.75), loc='upper center')
                ax_current_optimization_parameter.set_xlim(parameter_bounds[0][0], parameter_bounds[0][1])
                ax_current_optimization_parameter.set_ylim(parameter_bounds[1][0], parameter_bounds[1][1])
                ax_current_optimization_parameter.set_xlabel(str(list(model_hierarchy.parameters.keys())[0]))
                ax_current_optimization_parameter.set_ylabel(str(list(model_hierarchy.parameters.keys())[1]))

            parameter_optimization_widget.draw()

        list_of_reset_functions.append(reset_fig_parameter_optimization)

        scenarios.append(VBox([HBox([button_initialization_optimization, button_start_optimization,
                                     button_stop_optimization]),
                               HBox([label_current_objective_function_value, current_objective_function_value]),
                               HBox([label_current_optimization_parameter, current_optimization_parameter]),
                               parameter_optimization_widget]))
        scenarios_titles.append(translate('Parameter optimization'))
        scenarios_numbers['parameter_optimization'] = len(scenarios_numbers)

    ## Monte Carlo
    output_scalar = False
    global monte_carlo_counter
    monte_carlo_counter = 0
    if has_output:
        output = data['output'].ravel()
        if output.shape == (1,) or output_function:
            output_scalar = True
            if output_function:
                output = output_function(output, mu)

            mean_estimates = []

            if model_hierarchy.parameters.dim == 2:
                fig_output, axs = plt.subplots(1, 2)
                ax_output = axs[1]
                ax_monte_carlo_samples = axs[0]
                if model_hierarchy.parameters.dim == 2:
                    if density_function_monte_carlo:
                        x = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], 500)
                        y = np.linspace(parameter_bounds[1][0], parameter_bounds[1][1], 500)
                        X, Y = np.meshgrid(x, y)
                        pos = np.empty(X.shape + (2,))
                        pos[:, :, 0] = X
                        pos[:, :, 1] = Y
                        density = density_function_monte_carlo(pos)
                        colormap = plt.cm.get_cmap('viridis')
                        sm = plt.cm.ScalarMappable(cmap=colormap)
                        sm.set_clim(vmin=np.min(density), vmax=np.max(density))
                        fig_output.colorbar(sm, ax=ax_monte_carlo_samples)
            else:
                fig_output, ax_output = plt.subplots(1, 1)
            output_widget = fig_output.canvas

            def reset_fig_outputs():
                ax_output.clear()
                if model_hierarchy.parameters.dim == 2:
                    ax_monte_carlo_samples.clear()
                    for k, name in enumerate(model_names):
                        ax_monte_carlo_samples.scatter([], [], c=colors[k], label=name)
                    ax_monte_carlo_samples.set_xlim(parameter_bounds[0][0], parameter_bounds[0][1])
                    ax_monte_carlo_samples.set_ylim(parameter_bounds[1][0], parameter_bounds[1][1])
                    ax_monte_carlo_samples.set_xlabel(str(list(model_hierarchy.parameters.keys())[0]))
                    ax_monte_carlo_samples.set_ylabel(str(list(model_hierarchy.parameters.keys())[1]))
                    if density_function_monte_carlo:
                        ax_monte_carlo_samples.imshow(density, origin='lower', cmap='viridis',
                                                      vmin=np.min(density), vmax=np.max(density),
                                                      extent=(parameter_bounds[0][0], parameter_bounds[0][1],
                                                              parameter_bounds[1][0], parameter_bounds[1][1]))
                    #ax_monte_carlo_samples.legend(framealpha=1., bbox_to_anchor=(0.5, -0.75), loc='upper center')
                    if density_function_monte_carlo:
                        ax_monte_carlo_samples.set_title(translate('Randomly selected parameters and probability density function'))
                    else:
                        ax_monte_carlo_samples.set_title(translate('Randomly selected parameters'))
                ax_output.set_title(translate('Outputs and estimated statistics'))
                fig_output.canvas.header_visible = False
                fig_output.set_figwidth(fig_width)
                fig_output.set_figheight(fig_height)
                fig_output.tight_layout()
                fig_output.legends = []
                for k, name in enumerate(model_names):
                    ax_output.scatter([], [], c=colors[k], marker=marker_styles[0], label=name)
                ax_output.scatter([], [], c=colors[mod_num], marker=marker_styles[1])
                ax_output.plot([0], [0], c=colors[-1], label=translate('Estimated mean'))
                global line_mean_estimates
                line_mean_estimates = ax_output.plot([], [], c=colors[-1],
                                                     marker=marker_styles[1])[0]
                ax_output.legend(framealpha=1., ncols=4, bbox_to_anchor=(0, -0.15), loc='upper center')
                fig_output.subplots_adjust(bottom=0.225)
                output_widget.draw()
                global outputs
                outputs = []
                global inputs_to_outputs
                inputs_to_outputs = []
                global monte_carlo_counter
                monte_carlo_counter = 0

            list_of_reset_functions.append(reset_fig_outputs)

            button_start_monte_carlo = Button(description=translate('Start'), button_style='primary',
                                              layout=Layout(width='100px', height='50px'), disabled=False)
            button_stop_monte_carlo = Button(description=translate('Stop'), button_style='primary',
                                             layout=Layout(width='100px', height='50px'), disabled=True)
            label_current_number_samples = Label(f'{translate("Number of samples")}: ')
            label_current_estimated_mean = Label(f'{translate("Current estimated mean")}: ')
            label_current_estimated_variance = Label(f'{translate("Current estimated variance")}: ')
            current_number_samples = Label('0')
            current_estimated_mean = Label('-')
            current_estimated_variance = Label('-')
            variance_estimates = []
            global monte_carlo_running
            monte_carlo_running = False

            scenarios.append(VBox([HBox([button_start_monte_carlo, button_stop_monte_carlo]),
                                   HBox([label_current_number_samples, current_number_samples]),
                                   HBox([label_current_estimated_mean, current_estimated_mean]),
                                   HBox([label_current_estimated_variance, current_estimated_variance]),
                                   output_widget]))
            scenarios_titles.append(translate('Monte Carlo estimation'))
            scenarios_numbers['monte_carlo'] = len(scenarios_numbers)

    scenarios_accordion = Accordion(titles=[translate('Application scenarios')],
                                    children=[Accordion(titles=scenarios_titles,
                                                        children=scenarios, selected_index=0)],
                                    selected_index=0)

    if has_output and output_scalar:
        def run_monte_carlo(s_out):
            global monte_carlo_running
            while monte_carlo_running:
                if random_sampling_function is None:
                    mu = parameter_space.sample_randomly()
                else:
                    mu = model_hierarchy.parameters.parse(random_sampling_function())
                if parameter_space.contains(mu):
                    do_parameter_update(mu, s_out)
                else:
                    pass

        def do_start_monte_carlo(_):
            global monte_carlo_running
            if not monte_carlo_running:
                monte_carlo_running = True
            button_start_monte_carlo.disabled = True
            button_stop_monte_carlo.disabled = False
            if objective_function:
                button_start_optimization.disabled = True
                button_stop_optimization.disabled = True
                button_initialization_optimization.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['monte_carlo'],
                                                      f'{translate("Running")}: {translate("Monte Carlo estimation")}')

            from pymor.tools.random import spawn_rng
            thread_monte_carlo = threading.Thread(target=spawn_rng(run_monte_carlo),
                                                  args=(statistics_out, ))
            if not thread_monte_carlo.is_alive():
                thread_monte_carlo.start()

        def do_stop_monte_carlo(_):
            global monte_carlo_running
            if monte_carlo_running:
                monte_carlo_running = False
            button_start_monte_carlo.disabled = False
            button_stop_monte_carlo.disabled = True
            if objective_function:
                global optimization_initialized
                if optimization_initialized:
                    button_start_optimization.disabled = False
                button_stop_optimization.disabled = True
                button_initialization_optimization.disabled = False
            scenarios_accordion.children[0].set_title(scenarios_numbers['monte_carlo'],
                                                      translate('Monte Carlo estimation'))

        button_start_monte_carlo.on_click(do_start_monte_carlo)
        button_stop_monte_carlo.on_click(do_stop_monte_carlo)

    if objective_function:
        class OptimizationInterruptedError(Exception):
            pass

        collected_optimization_data = []

        global initial_guess
        initial_guess = mu.to_numpy()

        global optimization_interrupted
        optimization_interrupted = False

        parameter_bounds = np.array(tuple(np.array(b) for b in parameter_bounds))

        def run_optimization():
            from scipy.optimize import minimize as scipy_optimize
            global initial_guess
            global optimization_running
            while optimization_running:
                try:
                    optimization_results = scipy_optimize(partial(do_parameter_update, s_out=statistics_out),
                                                          x0=initial_guess, method=optimization_method,
                                                          bounds=parameter_bounds, options=optimization_options)
                    print(f"Optimization result: {optimization_results}")
                    do_stop_optimization(None)
                    break
                except OptimizationInterruptedError:
                    last_mu = collected_optimization_data[-1]['point']
                    initial_guess = last_mu.to_numpy()
                    optimization_running = False
                    break

        def do_initialization_optimization(_):
            global optimization_iterations
            optimization_iterations = 0
            global optimization_initialized
            optimization_initialized = True

            if initial_parameter:
                reset_fig_parameter_optimization()
                if model_hierarchy.parameters.dim == 2:
                    ax_current_optimization_parameter.scatter([initial_parameter.to_numpy()[0]],
                                                              [initial_parameter.to_numpy()[1]],
                                                              c='black', marker='x',
                                                              label=f'{translate("Initial guess")} {initial_parameter}')

                global initial_guess
                initial_guess = initial_parameter.to_numpy()
                global optimization_running
                optimization_running = True
                global optimization_interrupted
                optimization_interrupted = False
                do_parameter_update(initial_parameter, s_out=statistics_out)
                optimization_running = False
            button_start_optimization.disabled = False

        def do_start_optimization(_):
            global optimization_running
            if not optimization_running:
                optimization_running = True
                global optimization_interrupted
                optimization_interrupted = False
            button_initialization_optimization.disabled = True
            button_start_optimization.disabled = True
            button_stop_optimization.disabled = False
            if has_output and output_scalar:
                button_start_monte_carlo.disabled = True
                button_stop_monte_carlo.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['parameter_optimization'],
                                                      f'{translate("Running")}: {translate("Parameter optimization")}')

            from pymor.tools.random import spawn_rng
            thread_optimization = threading.Thread(target=spawn_rng(run_optimization))
            if not thread_optimization.is_alive():
                thread_optimization.start()

        def do_stop_optimization(_):
            global optimization_running
            if optimization_running:
                global optimization_interrupted
                optimization_interrupted = True
            button_initialization_optimization.disabled = False
            button_start_optimization.disabled = False
            button_stop_optimization.disabled = True
            if has_output and output_scalar:
                button_start_monte_carlo.disabled = False
                button_stop_monte_carlo.disabled = True
            scenarios_accordion.children[0].set_title(scenarios_numbers['parameter_optimization'],
                                                      translate('Parameter optimization'))

        button_initialization_optimization.on_click(do_initialization_optimization)
        button_start_optimization.on_click(do_start_optimization)
        button_stop_optimization.on_click(do_stop_optimization)

    right_pane.append(scenarios_accordion)

    right_pane = VBox(right_pane)
    right_pane.layout.width = '50%'
    left_pane = VBox(left_pane)
    left_pane.layout.width = '50%'
    general_style = HTML(
        '<style>.widget-label, .jp-Cell-outputArea label, .jp-RenderedHTMLCommon, .jupyter-button, '
        '.widget-readout, .jp-OutputArea-output.jp-RenderedHTMLCommon table {font-size: 1em !important;}'
        '.jp-Cell-outputArea {font-size: 1.5em !important;}'
        '.widget-play {font-size: 17.5px !important;}</style>',
        layout=Layout(display='none'),
    )
    widget = HBox([left_pane, right_pane, general_style])
    widget.layout.grid_gap = '2%'

    do_reset(None)

    def do_parameter_update(mu, s_out):
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
            U = model_hierarchy.reconstruct(U, mod_num)
            global current_sol
            current_sol = U
            visualizer.set(U[-1])
            if len(U) > 1:
                solution_visualizer_player.playing = False
                solution_time_step_slider.value = len(U)-1
            visualizer_widget.draw()
        if has_output and output_scalar:
            output = data['output'].ravel()
            if output_function:
                output = output_function(output, mu)

            global monte_carlo_running
            if monte_carlo_running:
                global outputs
                outputs.append(output)
                global monte_carlo_counter
                monte_carlo_counter = monte_carlo_counter + 1
                ax_output.scatter([monte_carlo_counter], [output], c=colors[mod_num], marker=marker_styles[1])
                if model_hierarchy.parameters.dim == 2:
                    ax_monte_carlo_samples.scatter([mu.to_numpy()[0]], [mu.to_numpy()[1]], c=colors[mod_num])
                low, high = ax_output.get_ylim()
                ax_output.set_ylim(min(low, output), max(high, output))

                current_number_samples.value = str(len(outputs))
                global inputs_to_outputs
                inputs_to_outputs.append(monte_carlo_counter)
                mean_estimate = np.mean(np.array(outputs), axis=0)
                mean_estimates.append(mean_estimate)
                current_estimated_mean.value = str(np.mean(np.array(outputs), axis=0))
                variance_estimate = np.var(np.array(outputs), axis=0)
                variance_estimates.append(variance_estimate)
                current_estimated_variance.value = str(np.var(np.array(outputs), axis=0))
                line_mean_estimates.set_data(np.array(inputs_to_outputs), np.array(mean_estimates))
                output_widget.draw()

            global manual_selection
            if manual_selection:
                global manual_selection_counter
                ax_parameter_selection_output_values.scatter([manual_selection_counter], [output],
                                                             c=colors[mod_num], marker=marker_styles[1])
                low, high = ax_parameter_selection_output_values.get_ylim()
                ax_parameter_selection_output_values.set_ylim(min(low, output), max(high, output))
                parameter_selection_output_values_widget.draw()


        for k, (est_err, name) in enumerate(zip(data['error_estimates'], model_names)):
            if est_err is not None:
                ax_error_estimates.scatter([global_counter], [est_err], c=colors[k])
        low, high = ax_error_estimates.get_ylim()
        arr_err_ests = np.array(data['error_estimates'])
        ax_error_estimates.set_ylim(min(low, np.min(arr_err_ests[arr_err_ests != np.array(None)]) * 0.9,
                                        model_hierarchy.get_tolerance() * 0.9),
                                    max(high, np.max(arr_err_ests[arr_err_ests != np.array(None)]) * 1.1,
                                        model_hierarchy.get_tolerance() * 1.1))

        global inputs_to_tolerances
        inputs_to_tolerances.append(global_counter)
        global tolerances
        tolerances.append(model_hierarchy.get_tolerance())
        global line_tolerances
        line_tolerances.set_data(np.array(inputs_to_tolerances), np.array(tolerances))
        error_estimates_widget.draw()

        for k, runtime in enumerate(data['runtimes']):
            if k > 0:
                ax_timings[0].bar(global_counter, runtime, bottom=np.sum(data['runtimes'][:k]), color=colors[k],
                                  edgecolor='black')
            else:
                ax_timings[0].bar(global_counter, runtime, color=colors[k], edgecolor='black')
        for k, training_time in enumerate(data['training_times']):
            ax_timings[1].bar(global_counter, training_time, color=colors[k], edgecolor='black', hatch='//')
        _, high_0 = ax_timings[0].get_ylim()
        _, high_1 = ax_timings[1].get_ylim()
        ax_timings[0].set_ylim(0., max(high_0, np.sum(data['runtimes']) * 1.1))
        ax_timings[1].set_ylim(0., max(high_1, np.sum(data['training_times']) * 1.1))
        timings_widget.draw()

        do_update_statistics(s_out)

        if objective_function:
            global optimization_running
            if optimization_running:
                global optimization_interrupted
                if optimization_interrupted:
                    raise OptimizationInterruptedError
                else:
                    global optimization_iterations
                    optimization_iterations = optimization_iterations + 1

                    mu = model_hierarchy.parameters.parse(mu)
                    output = model_hierarchy.output(mu).ravel()
                    quantity_of_interest = objective_function(output, mu)
                    current_objective_function_value.value = str(quantity_of_interest)
                    current_optimization_parameter.value = str(mu.to_numpy())

                    global num_tol
                    if model_hierarchy.parameters.dim == 2:
                        ax_current_optimization_parameter.scatter([mu.to_numpy()[0]], [mu.to_numpy()[1]],
                                                                  marker='.', c=colors[num_tol],
                                                                  label=f'{translate("Tolerance")}: {tols[-1]:.3e}')
                        handles, labels = ax_current_optimization_parameter.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax_current_optimization_parameter.legend(by_label.values(), by_label.keys(), framealpha=1., bbox_to_anchor=(0.5, -0.75), loc='upper center')
                        parameter_optimization_widget.draw()

                    ax_objective_functional_value.scatter([optimization_iterations], [quantity_of_interest],
                                                          c=colors[num_tol],
                                                          label=f'{translate("Tolerance")}: {tols[-1]:.3e}')
                    handles, labels = ax_objective_functional_value.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax_objective_functional_value.legend(by_label.values(), by_label.keys(), framealpha=1.)
                    low, high = ax_objective_functional_value.get_ylim()
                    ax_objective_functional_value.set_ylim(min(low, quantity_of_interest * 0.9),
                                                           max(high, quantity_of_interest * 1.1))
                    parameter_optimization_widget.draw()

                    collected_optimization_data.append({'point': mu, 'val': quantity_of_interest})
                    return quantity_of_interest

        return mod_num

    def do_manual_parameter_update(mu, s_out):
        global initial_guess
        initial_guess = mu.to_numpy()
        global manual_selection
        manual_selection = True
        global manual_selection_counter
        manual_selection_counter = manual_selection_counter + 1
        res = do_parameter_update(mu, s_out)
        manual_selection = False
        return res

    parameter_selector.on_change(partial(do_manual_parameter_update, s_out=statistics_out))

    return widget

    # TODO: Mention current status (solving, training, estimating, etc.) somewhere
