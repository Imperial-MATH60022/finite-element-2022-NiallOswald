# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    cube = np.indices([degree + 1] * cell.dim)[::-1]
    coords_sum = np.sum(cube, axis=0)
    return np.stack(cube, axis=-1)[coords_sum <= degree] / degree


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

    # Cast points to a np.ndarray
    points = np.array(points)

    # Construct the matrix of powers present in the Vandermonde matrix
    if cell is ReferenceInterval:
        i_p = np.arange(degree + 1).reshape(1, degree + 1)

    elif cell is ReferenceTriangle:
        x_p = np.concatenate([np.arange(d, -1, -1) for d in range(degree + 1)])
        y_p = np.concatenate([np.arange(d + 1) for d in range(degree + 1)])

        i_p = np.vstack([x_p, y_p])

    else:
        raise ValueError("Unknown reference cell")

    if grad:
        # Modify powers of the coordinates to account for the derivatives
        d_p = i_p[:, np.newaxis, :] - np.eye(cell.dim)[:, :, np.newaxis]
        # Repeat grid points into a new axis
        point_mat = np.repeat(points[:, :, np.newaxis], cell.dim, axis=2)
        # 'Outer-product'-like tensor power to compute all elements
        vand_grad = np.prod(point_mat[:, :, :, np.newaxis] ** d_p, axis=1)
        # Multiply by the powers of the coordinates to complete the derivatives
        vand_grad = np.einsum('ikj,kj->ijk', vand_grad, i_p, optimize=True)
        # Tidy up any NaNs or Infs
        vand_grad = np.nan_to_num(vand_grad, nan=0, posinf=0, neginf=0)

        return vand_grad

    # 'Outer-product'-like tensor power to compute all elements
    return np.prod(points[:, :, np.newaxis] ** i_p, axis=1)


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
            'ib...,bj->ij...',
            vandermonde_matrix(self.cell, self.degree, points, grad),
            self.basis_coefs,
            optimize=True
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

        return np.array([fn(x) for x in self.nodes])

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

        class Found(Exception):
            pass

        entity_nodes = {d: {e: [] for e in range(cell.entity_counts[d])}
                        for d in range(cell.dim + 1)}
        for i, n in enumerate(nodes):
            try:
                for d in range(cell.dim + 1):
                    for e in range(cell.entity_counts[d]):
                        if cell.point_in_entity(n, (d, e)):
                            entity_nodes[d][e].append(i)
                            raise Found
            except Found:
                pass

        super(LagrangeElement, self).__init__(cell, degree, nodes,
                                              entity_nodes)

class VectorFiniteElement:
    def __init__(self, element):
        """A vector finite element define over a cell.

        :param element: the :class:`~.finite_elements.FiniteElement`
            which underlies the vector finite element.
        """

        #: The :class:`~.finite_elements.FiniteElement`
        #: which underlies the vector finite element.
        self.element = element
        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = element.cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = element.degree

        self.dim = self.cell.dim

        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = {
            d: {
                e: [
                    i for n in element.entity_nodes[d][e]
                    for i in range(n, n + 2)
                ] for e in element.entity_nodes[d]
            }
            for d in element.entity_nodes
        }
        #: ``nodes_per_entity[d]`` is the number of entities
        #: associated with an entity of dimension d.
        self.nodes_per_entity = self.dim * element.nodes_per_entity

        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = [point for point in element.nodes for i in range(self.dim)]

        #: The list of basis coordinates corresponding to the nodes of the
        #: vector element.
        self.node_weights = np.tile(np.eye(self.dim), (len(self.nodes), 1))

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this vector finite element at the
        points provided.

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
        """

        # Tabulate the underlying scalar finite element
        scalar_table = self.element.tabulate(points, grad)

        # Reshape the table to match the vector finite element
        shape = list(scalar_table.shape)
        shape[1] *= self.dim
        shape.append(self.dim)
        vector_table = np.zeros(shape)

        # Populate the table for the vector finite element
        for d in range(self.dim):
            if grad:
                vector_table[:, d::self.dim, :, d] = scalar_table
            else:
                vector_table[:, d::self.dim, d] = scalar_table

        return vector_table
