import vaex.jupyter.traitlets as vt
import vaex
import numpy as np

def test_column_list_traitlets():
    df = vaex.from_scalars(x=1, y=2)
    df['z'] = df.x + df.y
    column_list = vt.ColumnsMixin(df=df)
    assert len(column_list.columns) == 3
    df['w'] = df.z * 2
    assert len(column_list.columns) == 4
    del df['w']
    assert len(column_list.columns) == 3


def test_expression():
    df = vaex.example()
    expression = df.widget.expression()
    assert expression.value is None
    expression.value = 'x'
    assert expression.value.expression == 'x'
    assert expression.valid
    expression.v_model = 'x+'
    assert not expression.valid

    expression = df.widget.expression(df.y)
    assert expression.value == 'y'


def test_column():
    df = vaex.example()
    column = df.widget.column()
    assert column.value is None
    column = df.widget.column(df.y)
    assert column.value == 'y'

