import vaex
import pytest
import numpy as np
import contextlib
import sklearn
import sklearn.preprocessing.data
import sklearn.preprocessing._encoders
import sklearn.preprocessing.data
import sklearn.preprocessing._encoders
import sklearn.decomposition.pca
# import sklearn.preprocessing.base
import numpy.linalg
import dask
import dask.array as da
from vaex.ml import relax_sklearn_check


@pytest.fixture
def df():
    x = np.arange(1, 11, dtype=np.float64)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    return df


@pytest.mark.parametrize("transpose", [True, False])
def test_basic(df, transpose):
    A = np.array(df)
    if transpose:
        A = A.T
        df = df.T
    assert A.shape == df.shape == df.numpy.shape
    assert A.dtype == df.dtype
    assert A.ndim == df.ndim == 2
    assert len(A) == len(A)


def test_binary_scalar(df):
    x = df.x.values
    y = df.y.values
    x2 = x + 3
    y2 = y + 3
    assert (df + 3).x.tolist() == x2.tolist()
    assert (df + 3).y.tolist() == y2.tolist()
    assert (df.T + 3).df.x.tolist() == x2.tolist()
    assert (df.T + 3).df.y.tolist() == y2.tolist()


def test_binary_array(df):
    A = np.array(df)
    x = df.x.values
    A2 = A.T + x
    df2 = (df.T + x).df
    assert np.array(df2).tolist() == A2.T.tolist()
    assert np.array(df2.T).tolist() == A2.tolist()


def test_aggregates(df):
    A = np.array(df)
    a = np.nanmax(A)
    passes = df.executor.passes
    assert a.tolist() == np.nanmax(df).tolist()
    assert df.executor.passes == passes + 1, "aggregation should be done in 1 pass"

    a = np.nanmax(A, axis=0)
    assert a.tolist() == np.nanmax(df, axis=0).tolist()
    a = np.nanmax(A.T, axis=1)
    assert a.tolist() == np.nanmax(df.T, axis=1).tolist()


@pytest.mark.xfail
def test_aggregates_columnwise(df):
    # similar to test_aggregates, but now it should go over the columns
    A = np.array(df)
    assert isinstance(np.nanmax(df, axis=1), vaex.Expression)
    assert a.tolist() == np.nanmax(df, axis=1).tolist()
    a = np.nanmax(A.T, axis=0)
    assert a.tolist() == np.nanmax(df.T, axis=0).tolist()


def test_zeros_like(df):
    z = np.zeros_like(df.x)
    assert z.tolist() == [0] * 10


def test_mean(df):
    means = np.mean(df)
    assert means[0] == df.x.mean()


def test_ufuncs(df):
    assert np.log(df).x.tolist() == df.x.log().tolist()
    assert np.log(df.T).df.x.tolist() == df.x.log().tolist()


def test_unary(df):
    assert (-df).x.tolist() == (-df.x.values).tolist()
    assert (np.negative(df)).x.tolist() == (-df.x.values).tolist()
    assert (-df.T).df.x.tolist() == (-df.x.values).tolist()
    assert (np.negative(df.T)).df.x.tolist() == (-df.x.values).tolist()


def test_dot(df):
    x = df.x.values
    y = df.y.values
    X = np.array(df)
    print(X.shape)
    Y = X.dot([[1, 0], [0, 1]])
    assert np.all(Y[:,0] == x)
    assert np.all(Y[:,1] == y)

    # not repeat with vaex
    with relax_sklearn_check(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1, 0], [0, 1]])
    Yv = np.array(df_dot)
    assert np.all(Yv[:,0] == x)
    assert np.all(Yv[:,1] == y)

    # check order
    Y = X.dot([[1, 1], [-1, 1]])
    assert np.all(Y[:,0] == x - y)
    assert np.all(Y[:,1] == y + x)

    with relax_sklearn_check(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1, 1], [-1, 1]])
    Yv = np.array(df_dot)
    assert np.all(Yv[:,0] == x - y)
    assert np.all(Yv[:,1] == y + x)

    # check non-square
    Y = X.dot([[1], [-1]])
    assert np.all(Y[:,0] == x - y)

    with relax_sklearn_check(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1], [-1]])
    Yv = np.array(df_dot)
    assert np.all(Yv[:,0] == x - y)



def test_sklearn_min_max_scalar(df):
    from sklearn.preprocessing import MinMaxScaler
    with relax_sklearn_check(), df.array_casting_disabled():
        scaler = MinMaxScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))

def test_sklearn_standard_scaler(df):
    from sklearn.preprocessing import StandardScaler
    with relax_sklearn_check(), df.array_casting_disabled():
        scaler = StandardScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))

@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("method", ['yeo-johnson', 'box-cox'])
def test_sklearn_power_transformer(df, standardize, method):
    from sklearn.preprocessing import PowerTransformer
    with relax_sklearn_check(), df.array_casting_disabled():
        power_trans_vaex = PowerTransformer(standardize=standardize, method=method, copy=True)
        dft = power_trans_vaex.fit_transform(df)
        assert isinstance(dft, vaex.DataFrame)

    X = np.array(df)
    power_trans_sklearn = PowerTransformer(standardize=standardize, method=method, copy=True)
    Xt = power_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)

@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_transformer(df, degree, interaction_only, include_bias):
    with relax_sklearn_check(), df.array_casting_disabled():
        poly_trans_vaex = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        dft = poly_trans_vaex.fit_transform(df)
        assert isinstance(dft, vaex.DataFrame)

    poly_trans_sklearn = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X = np.array(df)
    Xt = poly_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)

@pytest.mark.parametrize("output_distribution", ['uniform', 'normal'])
def test_quantile_transformer(df, output_distribution):
    with relax_sklearn_check(), df.array_casting_disabled():
        quant_trans_vaex = QuantileTransformer(n_quantiles=5, random_state=42, output_distribution=output_distribution)
        dft = quant_trans_vaex.fit_transform(df)
        assert isinstance(dft, vaex.DataFrame)

    quant_trans_sklearn = QuantileTransformer(n_quantiles=5, random_state=42, output_distribution=output_distribution)
    X = np.array(df)
    Xt = quant_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)

def test_sklearn_pca(df):
    from sklearn.decomposition import PCA
    for n in [1,2]:
        with relax_sklearn_check(), df.array_casting_disabled():
            scaler = PCA(n_components=n)
            scaler.fit(df)
            dfc = df.copy()
            dft = scaler.transform(dfc)
            assert isinstance(dft, vaex.DataFrame)
        X = np.array(df)
        Xt = scaler.transform(X)
        Xtv = np.array(dft)
        # we/dask do the math slighty different it seems, so not 100% the same
        numpy.testing.assert_almost_equal(Xt, Xtv)


def test_dask_qr(df):
    X = np.array(df)
    A, b = np.linalg.qr(X)
    Av, bv = np.linalg.qr(df)

    assert b.tolist() == bv.tolist()
    assert A.tolist() == Av.tolist()
