import collections
import time
import functools
import ipywidgets as widgets
from IPython.display import display, clear_output


def get_ioloop():
    import IPython
    import zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


debounced_execute_queue = []
debounce_enabled = True  # can be useful to turn off for debugging purposes


def _debounced_flush(recursive_counts=-1):
    """Run all non-executed debounced functions"""
    queue = debounced_execute_queue.copy()
    for f in queue:
        f()
        debounced_execute_queue.remove(f)
    if debounced_execute_queue and recursive_counts != 0:
        _debounced_flush(recursive_counts-1)


def debounced(delay_seconds=0.5, method=False):
    def wrapped(f):
        counters = collections.defaultdict(int)

        @functools.wraps(f)
        def execute(*args, **kwargs):
            if method:  # if it is a method, we want to have a counter per instance
                key = args[0]
            else:
                key = None
            counters[key] += 1

            @functools.wraps(execute)
            def debounced_execute(counter=counters[key]):
                if counter == counters[key]:  # only execute if the counter wasn't changed in the meantime
                    f(*args, **kwargs)
            if debounce_enabled:
                ioloop = get_ioloop()

                def thread_safe():
                    ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
                if ioloop is None:  # not in IPython
                    # debounced_execute()
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
