import vaex
import numpy as np
import vaex.jupyter
import xarray


class Grid:
    def __init__(self, df, models):
        self.df = df
        self.models = []
        self._callbacks_regrid = []
        self._callbacks_slice = []
        for model in models:
            self.model_add(model)
        # self.regrid()
        self.df.signal_selection_changed.connect(self.on_change_selection)

    def on_change_selection(self, df, name):
        # TODO: check name
        # print(name)
        # import traceback
        # traceback.print_stack()
        for model in self.models:
            model._prepare_regrid()
        self.regrid()

    def model_remove(self, model, regrid=True):
        index = self.models.index(model)
        del self.models[index]
        del self._callbacks_regrid[index]
        del self._callbacks_slice[index]

    def model_add(self, model):
        self.models.append(model)
        self._callbacks_regrid.append(model.signal_regrid.connect(self.regrid))
        self._callbacks_slice.append(model.signal_slice.connect(self.reslice))
        assert model.df == self.df

    def reslice(self, source_model=None):
        coords = []
        selections = self.models[0].selections
        selections = [k for k in selections if k is None or self.df.has_selection(k)]
        for model in self.models:
            subgrid = self.grid
            subgrid_sliced = self.grid
            axis_index = 1
            has_slice = False
            dims = ["selection"]
            coords = [selections.copy()]
            for other_model in self.models:
                if other_model == model:  # simply skip these axes
                    # for expression, shape, limit, slice_index in other_model.bin_parameters():
                    for axis in other_model.axes:
                        axis_index += 1
                        dims.append(str(axis.expression))
                        coords.append(axis.centers)
                else:
                    # for expression, shape, limit, slice_index in other_model.bin_parameters():
                    for axis in other_model.axes:
                        if axis.slice is not None:
                            subgrid_sliced = subgrid_sliced.__getitem__(tuple([slice(None)] * axis_index + [axis.slice])).copy()
                            subgrid = np.sum(subgrid, axis=axis_index)
                            has_slice = True
                        else:
                            subgrid_sliced = np.sum(subgrid_sliced, axis=axis_index)
                            subgrid = np.sum(subgrid, axis=axis_index)
            model.grid = xarray.DataArray(subgrid, dims=dims, coords=coords)
            if has_slice:
                model.grid_sliced = xarray.DataArray(subgrid_sliced)
            else:
                model.grid_sliced = None

    @vaex.jupyter.debounced(method=True, delay_seconds=0.05)
    def regrid(self, source_model=None):
        if not self.models:
            return
        binby = []
        shapes = []
        limits = []
        selections = self.models[0].selections
        for model in self.models:
            if model.selections != selections:
                raise ValueError('Selections for all models should be the same')
            # for expression, shape, limit, slice_index in model.bin_parameters():
            for axis in model.axes:
                binby.append(axis.expression)
                limits.append([axis.min, axis.max])
                shapes.append(axis.shape or axis.shape_default)
        selections = [k for k in selections if k is None or self.df.has_selection(k)]
        @vaex.delayed
        def assign_grid(grid):
            self.grid = grid
            self.reslice()
        assign_grid(self.df.count(binby=binby, shape=shapes, limits=limits, selection=selections, progress=self.progress, delay=True))
        self._execute()

    @vaex.jupyter.debounced(method=True, delay_seconds=0.05)
    def _execute(self):
        self.df.execute()

    def progress(self, f):
        return all([model.on_progress_grid(f) for model in self.models])
