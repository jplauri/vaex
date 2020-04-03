import collections
import time
import functools
import ipywidgets as widgets
from IPython.display import display, clear_output
import IPython
import asyncio


ipython = IPython.get_ipython()
debounced_execute_queue = []
_debounced_futures = []
debounce_enabled = True  # can be useful to turn off for debugging purposes
_is_gatherering = False


def get_ioloop():
    import zmq
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


def flush(recursive_counts=-1):
    """Run all non-executed debounced functions.

    If execution of debounced calls lead to scheduling of new calls, they will be recursively executed,
    with a limit or recursive_counts calls. recursive_counts=-1 means infinite.
    """
    queue = debounced_execute_queue.copy()
    for f in queue:
        f()
        debounced_execute_queue.remove(f)
    if debounced_execute_queue and recursive_counts != 0:
        flush(recursive_counts-1)


_debounced_flush = flush  # old alias, TODO: remove


async def gather(recursive_counts=-1):
    """Gather all debounced function result, useful for waiting till all schedules operations are executed
    """
    global _is_gatherering
    was_already_gatherering = _is_gatherering  # store old status
    _is_gatherering = True
    await asyncio.gather(*_debounced_futures)
    if _debounced_futures and recursive_counts != 0:
        await gather(recursive_counts-1)
    _is_gatherering = was_already_gatherering  # restore old status


def kernel_tick():
    """Execute a single command, to allow events from the frontend to get to the kernel during execution."""
    # For instance zoom events which should cancel vaex executions.
    # We should not execute more command during gathering, since that can execute the
    # next notebook cell. Maybe take a look at https://github.com/kafonek/ipython_blocking
    # for inspiration how to
    if ipython is not None and not _is_gatherering:
        ipython.kernel.do_one_iteration()


def debounced(delay_seconds=0.5, method=False, skip_gather=False):
    '''A decorator to debounce many method/function call into 1 call.

    Note: this only works in a IPython environment, such as a Jupyter notebook context. Outside
    of this context, calling :func:`flush` will execute pending calls.

    :param float delay_seconds: The amount of seconds that should pass without any call, before the (final) call will be executed.
    :param bool method: The decorator should know if the callable is a a method or not, otherwise the debounced is on a per-class basis.
    :param bool skip_gather: The decorated function will be be waited for when calling vaex.jupyter.gather()

    '''
    def wrapped(f):
        counters = collections.defaultdict(int)

        @functools.wraps(f)
        def execute(*args, **kwargs):
            if method:  # if it is a method, we want to have a counter per instance
                key = args[0]
            else:
                key = None
            counters[key] += 1
            future = asyncio.Future()
            if not skip_gather:
                _debounced_futures.append(future)

            @functools.wraps(execute)
            def debounced_execute(counter=counters[key]):
                try:
                    result = None
                    if counter == counters[key]:  # only execute if the counter wasn't changed in the meantime
                        result = f(*args, **kwargs)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
                finally:
                    if not skip_gather:
                        _debounced_futures.remove(future)
            if debounce_enabled:
                ioloop = get_ioloop()

                def thread_safe():
                    ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
                if ioloop is None:  # not in IPython
                    debounced_execute_queue.append(debounced_execute)
                    print("add to queue")
                else:
                    ioloop.add_callback(thread_safe)
            else:
                debounced_execute()
        execute.original = f
        return execute
    return wrapped


_selection_hooks = []


def interactive_cleanup():
    for dataset, f in _selection_hooks:
        dataset.signal_selection_changed.disconnect(f)


def interactive_selection(df):
    global _selection_hooks

    def wrapped(f_interact):
        if not hasattr(f_interact, "widget"):
            output = widgets.Output()

            def _selection_changed(df, selection_name):
                with output:
                    clear_output(wait=True)
                    f_interact(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            _selection_changed(df, None)
            display(output)
            return functools.wraps(f_interact)
        else:
            def _selection_changed(df, selection_name):
                f_interact.widget.update(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            return functools.wraps(f_interact)
    return wrapped
