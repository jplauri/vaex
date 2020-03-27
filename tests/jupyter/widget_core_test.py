import pytest
import vaex
import numpy as np
import vaex.jupyter.model
import vaex.jupyter.view
import vaex.jupyter.grid
from vaex.jupyter.utils import _debounced_flush as flush

@pytest.fixture()
def df():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = x ** 2
    g1 = np.array([0, 1, 1, 2, 2, 2])
    g2 = np.array([0, 0, 1, 1, 2, 3])
    ds = vaex.from_arrays(x=x, y=y, g1=g1, g2=g2)
    ds.categorize(ds.g1)
    ds.categorize(ds.g2)
    return ds


def test_axis_status(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression=df.x)
    assert x.status == x.Status.NO_LIMITS
    x._calculate_limits()
    assert x.status == x.Status.CALCULATING_LIMITS
    df.execute()
    assert x.status == x.Status.READY


def test_model_status(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='x', min=0, max=1)
    assert x.status == x.Status.READY
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    y = vaex.jupyter.model.Axis(df=df, expression='y')
    assert x.status == x.Status.NO_LIMITS
    assert y.status == y.Status.NO_LIMITS
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=y, shape=5)
    grid = vaex.jupyter.grid.Grid(df, [model])  # noqa
    assert x.status == x.Status.NO_LIMITS
    assert y.status == y.Status.NO_LIMITS
    assert model.status == model.Status.MISSING_LIMITS
    assert model.status_text == 'Missing limits for x, y'
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    flush(0)  # this will schedule the minmax calculations only
    # and it will schedule and execute task
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    assert x.status == x.Status.CALCULATING_LIMITS
    assert x.status == x.Status.CALCULATING_LIMITS
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for x, y'

    flush(0)  # this will execute the minmax (DataArrayModel._execute_tasks)
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    assert model.grid is None
    # this will trigger 4x DataArrayModel._update_grid (for min and max)
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 4
    flush(0)  # this will request the gridding
    assert model.status == model.Status.CALCULATING_GRID
    assert model.status_text == 'Aggregating data'
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    flush(1)  # this will schedule AND perform the gridding
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 0
    assert model.status == model.Status.READY
    assert model.status_text == 'Ready'
    assert model.grid is not None

    x.expression = df.x * 2
    # TODO: do we want a state 'DIRTY_LIMITS' ?
    flush(0)  # this will schedule the minmax calculations only for x
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for (x * 2)'
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1

    flush(0)  # this will execute the minmax (DataArrayModel._execute_tasks)
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    # this will trigger 1x DataArrayModel._update_grid (for x max, since 2*0=0)
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    flush(0)  # this will request the gridding
    assert model.status == model.Status.CALCULATING_GRID
    assert model.status_text == 'Aggregating data'
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    flush(1)  # this will schedule AND perform the gridding
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 0
    assert model.status == model.Status.READY
    assert model.status_text == 'Ready'
    assert model.grid is not None

    # this should trigger a recomputation
    df.select(df.x > 0)
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 1
    assert model.status == model.Status.CALCULATING_GRID
    assert model.status_text == 'Aggregating data'
    flush(1)  # this will schedule AND perform the gridding
    assert len(vaex.jupyter.utils.debounced_execute_queue) == 0


def test_histogram_model_passes(df, flush_guard):
    passes = df.executor.passes
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    model = vaex.jupyter.model.Histogram(df=df, x=x)
    assert df.executor.passes == passes
    flush()
    # this will do the minmax
    assert df.executor.passes == passes + 1

    # now will will manually do the grid
    grid = vaex.jupyter.grid.Grid(df, [model])
    grid.regrid()
    flush()
    assert df.executor.passes == passes + 2

    # a minmax and a new grid
    model.x.expression = 'y'
    assert df.executor.passes == passes + 2
    flush()
    assert df.executor.passes == passes + 2 + 2


def test_two_model_passes(df, flush_guard):
    passes = df.executor.passes
    x1 = vaex.jupyter.model.Axis(df=df, expression='x')
    x2 = vaex.jupyter.model.Axis(df=df, expression='x')
    model1 = vaex.jupyter.model.Histogram(df=df, x=x1)
    model2 = vaex.jupyter.model.Histogram(df=df, x=x2)
    assert df.executor.passes == passes
    flush()
    # this will do the minmax for both in 1 pass
    assert df.executor.passes == passes + 1

    # now we will manually do the gridding, both in 1 pass
    grid1 = vaex.jupyter.grid.Grid(df, [model1])
    grid2 = vaex.jupyter.grid.Grid(df, [model2])
    grid1.regrid()
    grid2.regrid()
    assert model1.grid is None
    assert model2.grid is None
    flush()
    assert df.executor.passes == passes + 1 + 1


def test_heatmap_model_passes(df, flush_guard):
    passes = df.executor.passes
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    y = vaex.jupyter.model.Axis(df=df, expression='y')
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=y, shape=5)
    assert df.executor.passes == passes
    flush()
    # this will do two minmaxes in 1 pass
    assert df.executor.passes == passes + 1

    # now will will manually do the grid
    grid = vaex.jupyter.grid.Grid(df, [model])
    grid.regrid()
    flush()
    assert df.executor.passes == passes + 2

    # once a minmax and a new grid
    x.expression = 'y'
    assert df.executor.passes == passes + 2
    flush()
    assert df.executor.passes == passes + 2 + 2

    # twice a minmax in 1 pass, followed by a gridding
    x.expression = 'x*2'
    y.expression = 'y*2'
    assert df.executor.passes == passes + 2 + 2
    flush()
    assert df.executor.passes == passes + 2 + 2 + 2

    # once a minmax and a new grid
    x.expression = 'x*3'
    assert df.executor.passes == passes + 2 + 2 + 2
    flush()
    assert df.executor.passes == passes + 2 + 2 + 2 + 2


def test_histogram_model(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='g1')
    model = vaex.jupyter.model.Histogram(df=df, x=x)
    grid = vaex.jupyter.grid.Grid(df, [model])
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 2.5
    assert model.x.shape == 3
    assert model.grid.data.tolist() == [[1, 2, 3]]
    assert model.grid.dims == ('selection', 'g1')
    assert model.grid.coords['selection'].data.tolist() == [None]
    assert model.grid.coords['g1'].data.tolist() == [0, 1, 2]

    viz = vaex.jupyter.view.Histogram(model=model, dimension_groups='slice')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3]]
    assert viz.plot.x_axis.label == 'g1'
    assert viz.plot.y_axis.label == 'count'

    model.x.expression = 'g2'
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 3.5
    assert model.x.shape == 4
    assert model.grid.data.tolist() == [[2, 2, 1, 1]]
    assert model.grid.dims == ('selection', 'g2')
    assert model.grid.coords['selection'].data.tolist() == [None]
    assert model.grid.coords['g2'].data.tolist() == [0, 1, 2, 3]
    assert viz.plot.x_axis.label == 'g2'

    x = vaex.jupyter.model.Axis(df=df, expression='x', min=-0.5, max=5.5)
    model = vaex.jupyter.model.Histogram(df=df, x=x, shape=6)
    flush()
    assert model.x.centers.tolist() == [0, 1, 2, 3, 4, 5]
    assert model.x.min == -0.5
    assert model.x.max == 5.5
    grid = vaex.jupyter.grid.Grid(df, [model])  # noqa
    assert model.x.shape is None
    assert model.shape == 6


def test_histogram_sliced(df, flush_guard):
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    g2 = vaex.jupyter.model.Axis(df=df, expression='g2')
    model1 = vaex.jupyter.model.Histogram(df=df, x=g1)
    model2 = vaex.jupyter.model.Histogram(df=df, x=g2)
    grid = vaex.jupyter.grid.Grid(df, [model1, model2])  # noqa
    flush()
    assert model1.x.centers.tolist() == [0, 1, 2]

    assert model1.grid.data.tolist() == [[1, 2, 3]]
    assert model2.grid.data.tolist() == [[2, 2, 1, 1]]

    viz = vaex.jupyter.view.Histogram(model=model1, dimension_groups='slice')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3]]
    assert model1.grid_sliced is None
    model2.x.slice = 0
    assert model1.grid.data.tolist() == [[1, 2, 3]]
    assert model1.grid_sliced.data.tolist() == [[1, 1, 0]]
    assert viz.plot.mark.y.tolist() == [[1, 2, 3], [1, 1, 0]]


def test_histogram_selections(df, flush_guard):
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    g2 = vaex.jupyter.model.Axis(df=df, expression='g2')
    df.select(df.g1 == 1)
    model1 = vaex.jupyter.model.Histogram(df=df, x=g1, selections=[None, True])
    model2 = vaex.jupyter.model.Histogram(df=df, x=g2, selections=[None, True])
    grid = vaex.jupyter.grid.Grid(df, [model1, model2])  # noqa
    flush()
    assert model1.grid.data.tolist() == [[1, 2, 3], [0, 2, 0]]
    assert model2.grid.data.tolist() == [[2, 2, 1, 1], [1, 1, 0, 0]]

    viz = vaex.jupyter.view.Histogram(model=model1, groups='selections')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3], [0, 2, 0]]


#  this should be tested in plot_widget_test
# def test_hist_controls(ds):
#     state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])

#     viz = vaex.jupyter.view.Histogram(state=state1)
#     assert viz.normalize == False
#     viz.control.normalize.v_model = True
#     assert viz.normalize

#     assert state1.x_expression == 'g1'
#     viz.control.x.v_model = 'g2'
#     assert state1.x_expression == 'g2'

# TODO: later enable this again?
# def test_geojson(ds):
#     geo_json = {
#         'features': [
#             {'geometry': {
#                 'type': 'MultiPolygon',
#                 'coordinates': []
#             },
#             'properties': {
#                 'objectid': 1
#             }
#             }
#         ]
#     }
#     state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
#     viz = vaex.jupyter.view.VizHistogramBqplot(state=state1, groups='slice')
#     vizgeo = vaex.jupyter.view.VizMapGeoJSONLeaflet(geo_json, ['g2'], state=state2)
#     assert state1.grid.tolist() == [[1, 2, 3]]
#     assert state2.grid.tolist() == [[2, 2, 1, 1]]
    
#     assert viz.bar.y.tolist() == [[1, 2, 3]]
#     assert state1.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [[1, 2, 3]]
#     assert state1.grid_sliced.tolist() == [[1, 1, 0]]
#     assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]


@pytest.mark.skip(reason='unsure why it does not work')
def test_piechart(ds, flush_guard):
    state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
    state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
    grid = vaex.jupyter.grid.Grid(ds, [state1, state2])  # noqa
    viz_pie = vaex.jupyter.view.VizPieChartBqplot(state=state1)
    viz_bar = vaex.jupyter.view.VizHistogramBqplot(state=state2)  # noqa
    assert state1.grid.tolist() == [1, 2, 3]
    assert state2.grid.tolist() == [2, 2, 1, 1]

    state2.x_slice = None
    assert viz_pie.pie1.sizes.tolist() == [1, 2, 3]
    assert state2.grid_sliced is None
    state2.x_slice = 0
    assert state1.grid.tolist() == [1, 2, 3]
    assert state1.grid_sliced.tolist() == [1, 1, 0]
    assert viz_pie.pie2.sizes.tolist() == [1, 1, 0]


def test_heatmap_model_basics(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='x', min=0, max=5)
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=g1, shape=2)
    grid = vaex.jupyter.grid.Grid(df, [model])
    flush()
    assert model.x.min == 0
    assert model.x.max == 5
    assert model.y.min == -0.5
    assert model.y.max == 2.5
    assert model.shape == 2
    assert model.x.shape is None
    assert model.y.shape == 3
    assert model.grid.data.tolist() == [[[1, 2, 0], [0, 0, 2]]]

    viz = vaex.jupyter.view.Heatmap(model=model)
    flush()
    # TODO: if we use bqplot-image-gl we can test the data again
    # assert viz.heatmap.color.T.tolist() == [[1, 2, 0], [0, 0, 2]]
    assert viz.plot.x_label == 'x'
    assert viz.plot.y_label == 'g1'

    model.x.expression = 'g2'
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 3.5
    assert model.x.shape == 4
    assert model.shape == 2
    grid = [[1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]]
    assert model.grid.data.tolist() == [grid]
    # TODO: if we use bqplot-image-gl we can test the data again
    # assert viz.heatmap.color.T.tolist() == grid

# def test_heatmap_sliced(ds):
#     state1 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state2.grid.tolist() == [2, 2, 1, 1]
    
#     viz = vaex.jupyter.view.VizHeatmapBqplot(state=state1)
#     assert viz.bar.y.tolist() == [1, 2, 3]
#     assert state1.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state1.grid_sliced.tolist() == [1, 1, 0]
#     assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]

# def test_heatmap_controls(ds):
#     state1 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='x')
#     state2 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='y')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])

#     viz = vaex.jupyter.view.VizHistogramBqplot(state=state1)
#     assert viz.normalize == False
#     viz.control.normalize.value = True
#     assert viz.normalize

#     assert state1.x_expression == 'g1'
#     viz.control.x.value = 'g2'
#     assert state1.x_expression == 'g2'


@pytest.mark.skip(reason='requires icons PR in ipywidgets')
def test_create(ds, flush_guard):
    creator = vaex.jupyter.create.Creator(ds)
    creator.widget_button_new_histogram.click()
    assert len(creator.widget_container.children) == 1
    assert len(creator.view) == 1
    assert creator.view[0].state.ds is ds
    # assert creator.widget_container.selected_index == 0
    assert len(creator.widget_buttons_remove) == 1

    creator.widget_button_new_heatmap.click()
    assert len(creator.widget_container.children) == 2
    assert len(creator.view) == 2
    assert creator.view[1].state.ds is ds
    # assert creator.widget_container.selected_index == 1
    assert len(creator.widget_buttons_remove) == 2

    creator.widget_button_new_histogram.click()
    # assert creator.widget_container.selected_index == 2
    assert len(creator.widget_buttons_remove) == 3
    assert len(creator.widget_container.children) == 3

    creator.widget_container.selected_index = 1
    creator.widget_buttons_remove[1].click()
    # assert creator.widget_container.selected_index == 1
    assert len(creator.widget_container.children) == 2
    assert len(creator.widget_buttons_remove) == 2
    assert len(creator.grid.states) == 2

    creator.widget_buttons_remove[1].click()
    # assert creator.widget_container.selected_index == 0
    assert len(creator.widget_container.children) == 1
    assert len(creator.widget_buttons_remove) == 1
    assert len(creator.grid.states) == 1
