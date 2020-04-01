import enum
import traitlets
import numpy as np
import xarray

import vaex
import vaex.jupyter
from .decorators import signature_has_traits
from .traitlets import Expression


@signature_has_traits
class Axis(traitlets.HasTraits):
    class Status(enum.Enum):
        NO_LIMITS = 1
        CALCULATING_LIMITS = 2
        READY = 3
    status = traitlets.UseEnum(Status, Status.NO_LIMITS)
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    expression = Expression()
    slice = traitlets.CInt(None, allow_none=True)
    min = traitlets.CFloat(None, allow_none=True)
    max = traitlets.CFloat(None, allow_none=True)
    centers = traitlets.Any()
    shape = traitlets.CInt(None, allow_none=True)
    shape_default = traitlets.CInt(64)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.min is not None and self.max is not None:
            self.status = Axis.Status.READY

    def __repr__(self):
        def myrepr(value, key):
            if isinstance(value, vaex.expression.Expression):
                return str(value)
            return value
        args = ', '.join('{}={}'.format(key, myrepr(getattr(self, key), key)) for key in self.traits().keys() if key != 'df')
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def has_missing_limit(self):
        # return not self.df.is_category(self.expression) and (self.min is None or self.max is None)
        return (self.min is None or self.max is None)

    def calculate_limits(self):
        """Force the calculation of min and max"""
        self._calculate_limits()
        self.df.execute()

    def _calculate_limits(self):
        categorical = self.df.is_category(self.expression)
        if categorical:
            N = self.df.category_count(self.expression)
            self.min, self.max = -0.5, N-0.5
            # centers = np.arange(N)
            # self.shape = N
            self._calculate_centers()
            self.status = Axis.Status.READY
        else:
            self.status = Axis.Status.CALCULATING_LIMITS

            def on_min_max(min_max):
                self.min, self.max = min_max
                self._calculate_centers()
                self.status = Axis.Status.READY

            def on_error():
                self.status = Axis.Status.NO_LIMITS

            task = self.df.minmax(self.expression, delay=True)
            return task.then(on_min_max, on_error)

    def _calculate_centers(self):
        categorical = self.df.is_category(self.expression)
        # if self.min is None or self.max is None:
        #     return # special condition that can occur during testing, since debounced does not work
        if categorical:
            N = self.df.category_count(self.expression)
            centers = np.arange(N)
            self.shape = N
        else:
            # print(self.expression, [min, max], getattr(self, attr + '_shape') or self.shape)
            centers = self.df.bin_centers(self.expression, [self.min, self.max], shape=self.shape or self.shape_default)
        self.centers = centers


@signature_has_traits
class DataArray(traitlets.HasTraits):
    class Status(enum.Enum):
        MISSING_LIMITS = 1
        CALCULATING_LIMITS = 2
        CALCULATING_GRID = 3
        READY = 3
    status = traitlets.UseEnum(Status, Status.MISSING_LIMITS)
    status_text = traitlets.Unicode('Initializing')
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    axes = traitlets.List(traitlets.Instance(Axis), [])
    grid = traitlets.Instance(xarray.DataArray, allow_none=True)
    grid_sliced = traitlets.Instance(xarray.DataArray, allow_none=True)
    shape = traitlets.CInt(64)
    selections = traitlets.List(traitlets.Union(
        [traitlets.Bool(), traitlets.Unicode(allow_none=True)]), [None])

    def __init__(self, **kwargs):
        super(DataArray, self).__init__(**kwargs)
        self.signal_slice = vaex.events.Signal()
        self.signal_regrid = vaex.events.Signal()
        self.signal_grid_progress = vaex.events.Signal()
        self.observe(lambda change: self.signal_regrid.emit(), 'selections')
        self._on_axis_status_change()

        # keep a set of axis that need new limits
        self._dirty_axes = set()
        for axis in self.axes:
            axis.observe(self._on_axis_status_change, 'status')
            traitlets.link((self, 'shape'), (axis, 'shape_default'))
            assert axis.df is self.df, "axes should have the same dataframe"
            axis.observe(lambda _: self.signal_slice.emit(self), ['slice'])
            axis.observe(lambda _: self._update_grid(), ['min', 'max', 'shape'])

            def closure(axis=axis):
                def calculate_limits_for_axis():
                    self._dirty_axes.add(axis)
                    self.calculate_limits(only_dirty=True)
                axis.observe(lambda _: calculate_limits_for_axis(), ['expression'])
            closure()
        if self.has_missing_limits:
            self.calculate_limits()
        else:
            for axis in self.axes:
                axis._calculate_centers()
            self._update_grid()

    def _on_axis_status_change(self, change=None):
        missing_limits = [axis for axis in self.axes if axis.status == Axis.Status.NO_LIMITS]
        calculating_limits = [axis for axis in self.axes if axis.status == Axis.Status.CALCULATING_LIMITS]

        def names(axes):
            return ", ".join([str(axis.expression) for axis in axes])

        if missing_limits:
            self.status = DataArray.Status.MISSING_LIMITS
            self.status_text = 'Missing limits for {}'.format(names(missing_limits))
        elif calculating_limits:
            self.status = DataArray.Status.CALCULATING_LIMITS
            self.status_text = 'Computing limits for {}'.format(names(calculating_limits))
        else:
            assert all([axis.status == Axis.Status.READY for axis in self.axes])
            # self.status_text = 'Computing limits for {}'.format(names(missing_limits))

    @property
    def has_missing_limits(self):
        return any([axis.has_missing_limit for axis in self.axes])

    def on_progress_grid(self, f):
        return all(self.signal_grid_progress.emit(f))

    @vaex.jupyter.debounced(method=True, delay_seconds=0.3)
    def calculate_limits(self, only_dirty=False):
        # TODO: we'd like to do this delayed (in 1 pass)
        axes = [axis for axis in self.axes if axis.has_missing_limit]
        if only_dirty:
            axes = self._dirty_axes.copy()
        tasks = []
        for axis in axes:
            self._dirty_axes.discard(axis)
            task = axis._calculate_limits()
            if task:
                tasks.append(task)
        if tasks:
            self._execute_tasks()

    @vaex.jupyter.debounced(method=True, delay_seconds=0.05)
    def _execute_tasks(self):
        self.df.execute()

    @vaex.jupyter.debounced(method=True, delay_seconds=0.1)
    def _update_grid(self):
        for axis in self.axes:
            axis._calculate_centers()
        self._prepare_regrid()
        self.signal_regrid.emit(None)

    def _prepare_regrid(self):
        self.status = DataArray.Status.CALCULATING_GRID
        self.status_text = 'Aggregating data'

    @traitlets.observe('grid')
    def _on_change_grid(self, change):
        self.status = DataArray.Status.READY
        self.status_text = 'Ready'


class Histogram(DataArray):
    x = traitlets.Instance(Axis)
    # type = traitlets.CaselessStrEnum(['count', 'min', 'max', 'mean'], default_value='count')
    # groupby = traitlets.Instance(Axis)
    # groupby_normalize = traitlets.Bool(False, allow_none=True)
    # grid = traitlets.Any()
    # grid_sliced = traitlets.Any()

    def __init__(self, **kwargs):
        kwargs['axes'] = [kwargs['x']]
        super().__init__(**kwargs)


class Heatmap(DataArray):
    x = traitlets.Instance(Axis)
    y = traitlets.Instance(Axis)

    def __init__(self, **kwargs):
        kwargs['axes'] = [kwargs['x'], kwargs['y']]
        super().__init__(**kwargs)
