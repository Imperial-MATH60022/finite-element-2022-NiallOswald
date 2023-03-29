'''Test tabulation of basis functions.'''
import pytest
from fe_utils import ReferenceTriangle, ReferenceInterval, LagrangeElement, VectorFiniteElement
import numpy as np


@pytest.mark.parametrize('cell', (ReferenceInterval, ReferenceTriangle))
def test_tabulate_type(cell):
    fe = LagrangeElement(cell, 2)

    fv = VectorFiniteElement(fe)

    points = np.ones((4, cell.dim))

    t = fv.tabulate(points)

    assert isinstance(t, np.ndarray), \
        "tabulate must return a numpy array, not a %s" % type(t)


@pytest.mark.parametrize('cell', (ReferenceInterval, ReferenceTriangle))
def test_tabulate_matrix_rank(cell):
    fe = LagrangeElement(cell, 2)

    fv = VectorFiniteElement(fe)

    points = np.ones((4, cell.dim))

    t = fv.tabulate(points)

    assert len(t.shape) == 3, \
        "tabulate must return a rank 3 array, not rank %s" % len(t.shape)


@pytest.mark.parametrize('cell, degree',
                         [(c, d)
                          for c in (ReferenceInterval, ReferenceTriangle)
                          for d in range(8)])
def test_tabulate_matrix_size(cell, degree):
    fe = LagrangeElement(cell, 2)

    fv = VectorFiniteElement(fe)

    points = np.ones((4, cell.dim))

    shape = fv.tabulate(points).shape

    correct_shape = (4, cell.dim * fe.nodes.shape[0], cell.dim)

    assert shape == correct_shape, \
        "tabulate should have returned an array of shape %s, not %s"\
        % (correct_shape, shape)


@pytest.mark.parametrize('cell, degree',
                         [(c, d)
                          for c in (ReferenceInterval, ReferenceTriangle)
                          for d in range(1, 8)])
def test_tabulate_at_nodes(cell, degree):
    """Check that tabulating at the nodes produces the identity matrix."""
    fe = LagrangeElement(cell, degree)

    fv = VectorFiniteElement(fe)

    for i in range(cell.dim):
        assert (np.round(fv.tabulate(fe.nodes)[:, i::cell.dim, i]-np.eye(len(fe.nodes)), 10) == 0).all()


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
