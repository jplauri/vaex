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
    x = np.arange(10, dtype=np.float64)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    return df

def test_zeros_like(df):
    z = np.zeros_like(df.x)
    assert z.tolist() == [0] * 10

def test_mean(df):
    means = np.mean(df)
    assert means[0] == df.x.mean()

def test_ufuncs(df):
    dfl = np.log(df)
    dfl.x.tolist() == df.x.log().tolist()

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

def test_sklearn_power_transformer(df):
    from sklearn.preprocessing import PowerTransformer
    with relax_sklearn_check(), df.array_casting_disabled():
        scaler = PowerTransformer(standardize=False)
        scaler.fit(df)

        dft = scaler.transform(df.copy())
        assert isinstance(dft, vaex.DataFrame)
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))


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
