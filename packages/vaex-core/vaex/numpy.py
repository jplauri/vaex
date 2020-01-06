import numpy as np

import vaex
from vaex import Expression


# implementing nep18: https://numpy.org/neps/nep-0018-array-function-protocol.html
_nep18_method_mapping = {}  # maps from numpy function to an Expression method
def nep18_method(numpy_function):
    def decorator(f):
        _nep18_method_mapping[numpy_function] = f
        return f
    return decorator


# # implementing nep13: https://numpy.org/neps/nep-0013-ufunc-overrides.html
_nep13_method_mapping = {}  # maps from numpy function to an Expression method
# def nep13_method(numpy_function):
#     def decorator(f):
#         _nep13_method_mapping[numpy_function] = f
#         return f
#     return decorator


def nep13_and_18_method(numpy_function):
    def decorator(f):
        _nep13_method_mapping[numpy_function] = f
        _nep18_method_mapping[numpy_function] = f
        return f
    return decorator


class DataFrameAccessorNumpy:
    def __init__(self, df, transposed=False):
        self.df = df
        self._transposed = transposed

    def __len__(self):
        return len(self.df)
    
    def __array__(self, dtype=None, parallel=True):
        ar = self.df.__array__(dtype=dtype, parallel=parallel)
        return ar.T if self._transposed else ar
    
    def __iter__(self):
        """Iterator over the column names."""
        if self._transposed:
            for name in self.df.get_column_names():
                yield self[name]
        else:
            raise ValueError("Iterating over rows is not supported")

    def __getitem__(self, item):
        return self.df.__getitem__(item)

    @property
    def numpy(self):
        return self

    @property
    def T(self):
        return type(self)(self.df, transposed=not self._transposed)

    @property
    def shape(self):
        if self._transposed:
            return (len(self.df.get_column_names()), len(self))
        else:
            return (len(self), len(self.df.get_column_names()))

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        dtypes = [self[k].dtype for k in self.df.get_column_names()]
        assert all([dtypes[0] == dtype for dtype in dtypes])
        return dtypes[0]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        method = _nep13_method_mapping.get(ufunc)
        if method is None:
            return NotImplemented
        # if len(inputs) > 1 and inputs[1] is self:
        #     assert len(inputs) == 2  # TODO: check if arguments can be swapped? how?
        #     inputs = inputs[::-1]
        # if inputs[0] is not self:
        #     return NotImplemented
        # assert inputs[0] is self or inputs[1] is self
        return method(*inputs)

    def __array_function__(self, func, types, args, kwargs):
        method = _nep18_method_mapping.get(func)
        if method is None:
            return NotImplemented

        assert args[0] is self.df or args[0] is self
        result = method(*args, **kwargs)
        return result


    @nep18_method(np.mean)
    def _np_mean(self, axis=None):
        assert axis in [0, None]
        return self.mean(self.get_column_names())

    @nep18_method(np.dot)
    def _dot(self, b):
        b = np.asarray(b)
        assert b.ndim == 2
        N = b.shape[1]
        df = self.copy()
        names = df.get_column_names()
        output_names = ['c'+str(i) for i in range(N)]
        columns = [df[names[j]] for j in range(b.shape[0])]
        for name in names:
            df._hide_column(name)
        for i in range(N):
            def dot_product(a, b):
                products = ['%s * %s' % (ai, bi) for ai, bi in zip(a, b)]
                return ' + '.join(products)
            df[output_names[i]] = dot_product(columns, b[:,i])
        return df

    @nep18_method(np.may_share_memory)
    def _may_share_memory(self, b):
        return True  # be conservative

    @nep18_method(np.linalg.svd)
    def _np_linalg_svd(self, full_matrices=True):
        import dask.array as da
        import dask
        X = self.to_dask_array()
        # TODO: we ignore full_matrices
        u, s, v = da.linalg.svd(X)#, full_matrices=full_matrices)
        u, s, v = dask.compute(u, s, v)
        return u, s, v

    @nep18_method(np.linalg.qr)
    def _np_linalg_qr(self):
        import dask.array as da
        import dask
        X = self.to_dask_array()
        result = da.linalg.qr(X)
        result = dask.compute(*result)
        return result


for op in vaex.expression._binary_ops:
    name = op.get('numpy_name', op['name'])
    if name in ['contains', 'is', 'is_not']:
        continue
    numpy_function = getattr(np, name)
    assert numpy_function, 'numpy does not have {}'.format(name)
    def closure(name=name, numpy_function=numpy_function):
        def binary_op_method(self, rhs):
            assert isinstance(self, DataFrameAccessorNumpy)
            df = self.df.copy()
            if isinstance(rhs, np.ndarray):
                if self._transposed:
                    name = vaex.utils.find_valid_name('__aux', used=df.get_column_names(hidden=True))
                    df.add_column(name, rhs)
                    rhs = df[name]
                    for i, name in enumerate(self.df.get_column_names()):
                        df[name] = numpy_function(df[name], rhs)
                else:
                    for i, name in enumerate(self.df.get_column_names()):
                        df[name] = numpy_function(df[name], rhs[i])
            else:
                for i, name in enumerate(self.df.get_column_names()):
                    df[name] = numpy_function(df[name], rhs)
            if self._transposed:
                return df.T
            else:
                return df
        return binary_op_method

    # this implements e.g. numpy.multiply
    nep13_and_18_method(numpy_function)(closure())

    # while this implements the __mul__ method
    def closure2(numpy_function=numpy_function):
        def f(a, b):
            return numpy_function(a, b)
        return f

    dundername = '__{}__'.format(name)
    setattr(DataFrameAccessorNumpy, dundername, closure2())


for op in vaex.expression._unary_ops:
    name = op['name']
    numpy_name = op.get('numpy_name', name)
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(name)
    def closure(name=name, numpy_function=numpy_function):
        def unary_op_method(self):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
                self = DataFrameAccessorNumpy(self)
            df = df.copy()
            for i, name in enumerate(df.get_column_names()):
                df[name] = numpy_function(df[name])
            if self._transposed:
                return df.T
            else:
                return df
        return unary_op_method
    nep13_and_18_method(numpy_function)(closure())

    def closure2(numpy_function=numpy_function):
        def f(a):
            return numpy_function(a)
        return f

    dundername = '__{}__'.format(name)
    setattr(DataFrameAccessorNumpy, dundername, closure2())


for name, numpy_name in vaex.functions.numpy_function_mapping + [('isnan', 'isnan')]:
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(numpy_name)
    def closure(name=name, numpy_name=numpy_name, numpy_function=numpy_function):
        def forward_call(self, *args, **kwargs):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
                self = df.numpy
            # assert isinstance(self, DataFrameAccessorNumpy)
            df = df.copy()
            # if isinstance(rhs, np.ndarray):
            #     if self._transposed:
            #         name = vaex.utils.find_valid_name('__aux', used=df.get_column_names(hidden=True))
            #         df.add_column(name, rhs)
            #         rhs = df[name]
            #         for i, name in enumerate(self.df.get_column_names()):
            #             df[name] = numpy_function(df[name], rhs)
            #     else:
            #         for i, name in enumerate(self.df.get_column_names()):
            #             df[name] = numpy_function(df[name], rhs[i])
            # else:
            for name in df.get_column_names():
                df[name] = numpy_function(df[name], *args, **kwargs)
            if self._transposed:
                return df.T
            else:
                return df
        return forward_call

    print("add", numpy_function)
    nep13_and_18_method(numpy_function)(closure())




aggregates_functions = [
    'nanmin',
    'nanmax',
    'nansum',
    'nanvar',
    'sum',
]

for numpy_name in aggregates_functions:
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(numpy_name)
    def closure(numpy_name=numpy_name, numpy_function=numpy_function):
        def forward_call(self, *args, **kwargs):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
            results = []
            forward_kwargs = kwargs.copy()
            if 'axis' in kwargs:
                if kwargs['axis'] == 0 and not self._transposed:
                    pass  # this is fine
                elif kwargs['axis'] == 1 and self._transposed:
                    forward_kwargs['axis'] = 0  # since we are transposed we need to change this axis
                else:
                    raise ValueError("not supported: numpy.%s with kwargs %r" % (numpy_name, kwargs))                
            for name in df.get_column_names():
                method = vaex.expression._nep18_method_mapping[numpy_function]
                results.append(method(*(df[name],) + args, **forward_kwargs, delay=True))
            df.execute()
            results = [k.get() for k in results]
            # TODO: support axis argument
            results = np.array(results)
            if 'axis' in kwargs:
                return results
            return numpy_function(results)
        return forward_call

    nep13_and_18_method(numpy_function)(closure())

