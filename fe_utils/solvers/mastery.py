"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import sin, cos, sqrt, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs_u, fs_p, f):
    """Assemble the finite element system for the Stokes problem given
    the function space in which to solve and the right hand side
    function."""

    fv = fs_u.element
    fe = fs_p.element
    mesh = fs_u.mesh

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2 * fv.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi_u = fv.tabulate(Q.points)
    phi_p = fe.tabulate(Q.points)
    grad_phi_u = fv.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    A = sp.lil_matrix((fs_u.node_count, fs_u.node_count))
    B = sp.lil_matrix((fs_p.node_count, fs_u.node_count))
    C = sp.lil_matrix((fs_p.node_count, fs_p.node_count))
    l = np.zeros(fs_u.node_count + fs_p.node_count)  # noqa: E741

    # Assemble the matrix equation.
    for c in range(mesh.entity_counts[-1]):
        J = mesh.jacobian(c)
        invJ = np.linalg.inv(J)
        detJ = abs(np.linalg.det(J))

        nodes_u = fs_u.cell_nodes[c, :]
        nodes_p = fs_p.cell_nodes[c, :]

        eps_u = (np.einsum("ak,qial->qikl", invJ, grad_phi_u)
                 + np.einsum("al,qiak->qikl", invJ, grad_phi_u)) / 2
        eps_phi_u = (np.einsum("ak,qjal->qjkl", invJ, grad_phi_u)
                     + np.einsum("al,qjak->qjkl", invJ, grad_phi_u)) / 2

        A[np.ix_(nodes_u, nodes_u)] += np.einsum(
            "qikl,qjkl,q->ij", eps_u, eps_phi_u, Q.weights, optimize=True
        ) * detJ

        B[np.ix_(nodes_p, nodes_u)] -= np.einsum(
            "qi,ka,qjka,q->ij", phi_p, invJ, grad_phi_u, Q.weights, optimize=True
        ) * detJ

        l[nodes_u] += np.einsum(
            "k,qkj,qij,q->i", f.values[nodes_u], phi_u, phi_u, Q.weights, optimize=True
        ) * detJ

    # Insert boundary conditions
    boundary_u = boundary_nodes(fs_u)

    A[boundary_u, :] = 0
    A[boundary_u, boundary_u] = 1
    l[boundary_u] = 0

    B[0, :] = 0
    C[0, 0] = 1
    l[fs_u.node_count] = 0

    L = sp.bmat([[A, B.T], [B, C]])

    return L, l


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return (1., 1.)
        else:
            return (0., 0.)

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given
    resolution. It should return both the solution
    :class:`~fe_utils.function_spaces.Function` and the :math:`L^2` error in
    the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, 1)
    fv = VectorFiniteElement(LagrangeElement(mesh.cell, 2))
    fs_u = FunctionSpace(mesh, fv)
    fs_p = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer_u = Function(fs_u)
    analytic_answer_p = Function(fs_p)
    analytic_answer_u.interpolate(lambda x: (
        2*pi*sin(2*pi*x[1])*(1 - cos(2*pi*x[0])),
        -2*pi*sin(2*pi*x[0])*(1 - cos(2*pi*x[1]))
    ))
    analytic_answer_p.interpolate(lambda x: 0)


    # If the analytic answer has been requested then bail out now.
    if analytic:
        return (analytic_answer_u, analytic_answer_p), 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs_u)
    d3fdx3 = lambda x: - 8 * pi**3 * sin(2 * pi * x[0]) * (1 - cos(2 * pi * x[1]))
    d3fdx2y = lambda x: 8 * pi**3 * cos(2 * pi * x[0]) * sin(2 * pi * x[1])
    d3fdxy2 = lambda x: 8 * pi**3 * cos(2 * pi * x[1]) * sin(2 * pi * x[0])
    d3fdy3 = lambda x: - 8 * pi**3 * sin(2 * pi * x[1]) * (1 - cos(2 * pi * x[0]))
    f.interpolate(lambda x: (-0.5 * d3fdx2y(x) - 0.5 * d3fdy3(x),
                             0.5 * d3fdxy2(x) + 0.5 * d3fdx3(x)))

    # Assemble the finite element system.
    A, l = assemble(fs_u, fs_p, f)

    # Create the function to hold the solution.
    u = Function(fs_u)
    p = Function(fs_p)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csc_matrix(A)
    A_fac = splinalg.splu(A)
    solution = A_fac.solve(l)
    u.values[:] = solution[:fs_u.node_count]
    p.values[:] = solution[fs_u.node_count:]

    # Compute the L^2 error in the solution for testing purposes.
    error = sqrt(vector_errornorm(analytic_answer_u, u)**2
                 + errornorm(analytic_answer_p, p)**2)

    if return_error:
        u.values -= analytic_answer_u.values
        p.values -= analytic_answer_p.values

    return (u, p), error


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument(
        "--analytic", action="store_true",
        help="Plot the analytic solution instead of solving the finite"
        " element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument(
        "resolution", type=int, nargs=1,
        help="The number of cells in each direction on the mesh."
    )
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    (u, p), error = solve_mastery(resolution, analytic, plot_error)

    u.plot()
