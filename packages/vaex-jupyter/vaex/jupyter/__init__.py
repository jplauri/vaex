# -*- coding: utf-8 -*-
import os
import logging
import time
from .utils import debounced, interactive_selection, interactive_cleanup  # noqa
import vaex
import IPython.display


base_path = os.path.dirname(__file__)
logger = logging.getLogger("vaex.jupyter")


def _add_toolbar(viz):
    from .widgets import ToolsToolbar
    from traitlets import link
    toolbar = ToolsToolbar(supports_transforms=viz.supports_transforms, supports_normalize=viz.supports_normalize)
    viz.children = [toolbar, ] + viz.children
    link((viz, 'tool'), (toolbar, 'interact_value'))
    link((viz, 'transform'), (toolbar, 'transform_value'))
    return toolbar


class DataFrameAccessorWidget(object):
    def __init__(self, df):
        self.df = df
        import vaex.jupyter.grid
        self.grid = vaex.jupyter.grid.Grid(df, [])

    def clear(self):
        self.grid = vaex.jupyter.grid.Grid(self.df, [])

    def data_array(self, axes=[], selections=[None, True], shared=False, display_function=IPython.display.display, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        selections = selections.copy()
        model = vaex.jupyter.model.DataArrayModel(df=self.df, axes=axes, selections=selections, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.model_add(model)
        view = vaex.jupyter.view.DataArray(model=model, display_function=display_function)
        return view

    def _axes(self, expressions, limits):
        limits = self.df.limits(expressions, limits)
        axes = [vaex.jupyter.model.Axis(df=self.df, expression=expression, min=min, max=max) for expression, (min, max) in zip(expressions, limits)]
        return axes

    def histogram(self, x, limits=None, selections=[None, True], toolbar=True, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        selections = selections.copy()
        x, = self._axes([x], limits)
        model = vaex.jupyter.model.Histogram(df=self.df, x=x, selections=selections, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.model_add(model)
        viz = vaex.jupyter.view.Histogram(model=model)
        if toolbar:
            viz.toolbar = _add_toolbar(viz)
        return viz

    def pie(self, x, limits=None, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        x, = self._axes([x], limits)
        model = vaex.jupyter.model.Histogram(df=self.df, x=x, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.model_add(model)
        viz = vaex.jupyter.view.PieChart(model=model)
        return viz

    def heatmap(self, x, y, limits=None, selections=[None, True], transform='log', toolbar=True, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        x, y = self._axes([x, y], limits)
        model = vaex.jupyter.model.Heatmap(df=self.df, x=x, y=y, selections=selections, shape=256, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.model_add(model)
        viz = vaex.jupyter.view.Heatmap(model=model, transform=transform)
        if toolbar:
            viz.toolbar = _add_toolbar(viz)
        return viz

    def expression(self, initial_value=None):
        from .widgets import ExpressionTextArea
        return ExpressionTextArea(df=self.df, v_model=str(initial_value) if initial_value is not None else None)

    def column(self, initial_value=None):
        from .widgets import ColumnPicker
        return ColumnPicker(df=self.df, value=str(initial_value) if initial_value is not None else None)

    def selection(self, initial_value, name='default'):
        from .widgets import ExpressionSelectionTextArea
        return ExpressionSelectionTextArea(df=self.df, selection_name=name, v_model=str(initial_value) if initial_value is not None else None)

    def progress_circular(self, width=10, size=70, color='#82B1FF', text='', auto_hide=False):
        from .widgets import ProgressCircularNoAnimation
        progress_circular = ProgressCircularNoAnimation(width=width, size=size, color=color, text=text, value=0)

        @self.df.executor.signal_begin.connect
        def progress_begin():
            if auto_hide:
                progress_circular.hidden = False

        @self.df.executor.signal_progress.connect
        def update_progress(value):
            progress_circular.value = value*100
            return True

        @self.df.executor.signal_end.connect
        def progress_update():
            if auto_hide:
                progress_circular.hidden = True
        return progress_circular

    def counter_processed(self, postfix="rows processed", update_interval=0.2):
        from .widgets import Counter
        counter_processed = Counter(value=0, postfix=postfix)
        last_time = 0

        @self.df.executor.signal_begin.connect
        def progress_begin():
            nonlocal last_time
            last_time = time.time()

        @self.df.executor.signal_progress.connect
        def update_progress(value):
            nonlocal last_time
            number = int(value * len(self.df))
            current_time = time.time()
            if (current_time - last_time) > update_interval or value in [0, 1]:
                counter_processed.value = number
                last_time = current_time
            return True

        return counter_processed

    def counter_selection(self, selection, postfix="rows selected", update_interval=0.2, lazy=False):
        from .widgets import Counter
        selected = self.df.count(selection=selection).item() if self.df.has_selection(name=selection) else 0
        counter_selected = Counter(value=selected, postfix=postfix)

        dirty = False
        @self.df.signal_selection_changed.connect
        def selection_changed(df, name):
            nonlocal dirty
            if name == selection:
                # we only need to run once
                if not dirty:
                    dirty = True

                    def update_value(value):
                        nonlocal dirty
                        dirty = False
                        try:
                            value = value.item()
                        except:  # noqa
                            pass
                        counter_selected.value = value
                    # if lazy is True, this will only schedule the calculation, not yet execute it
                    if lazy:
                        vaex.delayed(update_value)(self.df.count(selection=selection, delay=True))
                    else:
                        update_value(self.df.count(selection=selection))

        return counter_selected
    #     from .widgets import Tools
    #     from traitlets import link
    #     viz = [] if viz is None else viz
    #     viz = [viz] if not isinstance(viz, (tuple, list)) else viz
    #     tools = Tools(value=initial_value, children=[k.widget for k in viz])
    #     for v in viz:
    #         link((tools, 'value'), (v, 'tool'))
    #     return tools

    # def card(plot, title=None, subtitle=None, **kwargs):
    #     from .widget import Card
    #     return Card(main=plot, title=title, subtitle,


def add_namespace():
    pass
