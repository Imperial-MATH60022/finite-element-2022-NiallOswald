# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import (ReferenceCell, ReferenceInterval,
                                 ReferenceTriangle)
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell: ReferenceCell, degree: int) -> np.ndarray:
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    cube = np.indices([degree + 1] * cell.dim)
    coords_sum = np.sum(cube, axis=0)
    return np.vstack([
        grid[coords_sum <= degree] for grid in cube]).T / degree


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    # Construct the matrix of powers of the coordinates
    if cell is ReferenceInterval:
        i_p = np.array([np.arange(degree + 1)])

    elif cell is ReferenceTriangle:
        x_p = np.concatenate([np.arange(d, -1, -1) for d in range(degree + 1)])
        y_p = np.concatenate([np.arange(d + 1) for d in range(degree + 1)])

        i_p = np.vstack([x_p, y_p])

    else:
        raise NotImplementedError

    if grad:
        d_p = i_p - np.eye(cell.dim).reshape((cell.dim, cell.dim, 1))

        vand_grad = np.array(
            [np.prod([points[:, i].reshape(-1, 1) ** p
                      for i, p in enumerate(c_p)], axis=0) for c_p in d_p]
        )
        vand_grad = np.einsum('ijk,ik->ijk', vand_grad, i_p)
        vand_grad = np.nan_to_num(vand_grad, nan=0, posinf=0, neginf=0)

        # Permute axes to match notes
        return vand_grad.transpose(1, 2, 0)

    return np.prod([points[:, i].reshape(-1, 1) ** p
                    for i, p in enumerate(i_p)], axis=0)


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            point evaluation node locations on the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with
            entity `(d, i)` of dimension `d` and index `i`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        self.basis_coefs = np.linalg.inv(
            vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """

        return np.einsum(
            'ib,bj',
            vandermonde_matrix(self.cell, self.degree, points, grad),
            self.basis_coefs
        )

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        raise NotImplementedError

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        super(LagrangeElement, self).__init__(cell, degree, nodes)
