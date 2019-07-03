# -*- coding: UTF-8 -*-

import numpy as np

from sympy import pi, cos, sin
from sympy import S

from sympde.core     import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div

from sympde.topology import dx, dy, dz
from sympde.topology import ScalarField
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import element_of_space
from sympde.topology import Domain
from sympde.topology import Boundary, trace_0, trace_1
from sympde.expr     import BilinearForm, LinearForm
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.topology import Domain, Line, Square, Cube

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL

import time
from tabulate import tabulate
from collections import namedtuple
from mpi4py import MPI

Timing = namedtuple('Timing', ['kind', 'time'])

DEBUG = False

# Communicator, size, rank
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

#def test_api_vector_poisson_2d():
#    print('============ test_api_vector_poisson_2d =============')
#
#    # ... abstract model
#    U = VectorFunctionSpace('U', domain)
#    V = VectorFunctionSpace('V', domain)
#
#    x,y = domain.coordinates
#
#    F = VectorField(V, name='F')
#
#    v = VectorTestFunction(V, name='v')
#    u = VectorTestFunction(U, name='u')
#
#    expr = inner(grad(v), grad(u))
#    a = BilinearForm((v,u), expr)
#
#    f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y), 2*pi**2*sin(pi*x)*sin(pi*y))
#
#    expr = dot(f, v)
#    l = LinearForm(v, expr)
#
#    # TODO improve
#    error = F[0] -sin(pi*x)*sin(pi*y) + F[1] -sin(pi*x)*sin(pi*y)
#    l2norm = Norm(error, domain, kind='l2', name='u')
#    # ...
#
#    # ... discrete spaces
##    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
#    Vh = ProductFemSpace(Vh, Vh)
#    # ...
#
#    # ...
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    M_f90 = ah.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (matrix) = ', te-tb)
#    t_f90 = te-tb
#
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    M_py = ah.assemble()
#    te = time.time()
#    print('> [python] elapsed time (matrix) = ', te-tb)
#    t_py = te-tb
#
#    matrix_timing = Timing('matrix', t_py, t_f90)
#    # ...
#
#    # ...
#    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    L_f90 = lh.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (rhs) = ', te-tb)
#    t_f90 = te-tb
#
#    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    L_py = lh.assemble()
#    te = time.time()
#    print('> [python] elapsed time (rhs) = ', te-tb)
#    t_py = te-tb
#
#    rhs_timing = Timing('rhs', t_py, t_f90)
#    # ...
#
#    # ... coeff of phi are 0
#    phi = VectorFemField( Vh, 'phi' )
#    # ...
#
#    # ...
#    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    L_f90 = l2norm_h.assemble(F=phi)
#    te = time.time()
#    t_f90 = te-tb
#    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)
#
#    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    L_py = l2norm_h.assemble(F=phi)
#    te = time.time()
#    print('> [python] elapsed time (L2 norm) = ', te-tb)
#    t_py = te-tb
#
#    l2norm_timing = Timing('l2norm', t_py, t_f90)
#    # ...
#
#    # ...
#    print_timing([matrix_timing, rhs_timing, l2norm_timing])
#    # ...
#
#def test_api_stokes_2d():
#    print('============ test_api_stokes_2d =============')
#
#    # ... abstract model
#    V = VectorFunctionSpace('V', domain)
#    W = FunctionSpace('W', domain)
#
#    v = VectorTestFunction(V, name='v')
#    u = VectorTestFunction(V, name='u')
#    p = ScalarTestFunction(W, name='p')
#    q = ScalarTestFunction(W, name='q')
#
#    A = BilinearForm((v,u), inner(grad(v), grad(u)), name='A')
#    B = BilinearForm((v,p), div(v)*p, name='B')
#    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q), name='a')
#    # ...
#
#    # ... discrete spaces
##    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
#
#    # TODO improve this?
#    Vh = ProductFemSpace(Vh, Vh, Vh)
#    # ...
#
#    # ...
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    M_f90 = ah.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (matrix) = ', te-tb)
#    t_f90 = te-tb
#
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    M_py = ah.assemble()
#    te = time.time()
#    print('> [python] elapsed time (matrix) = ', te-tb)
#    t_py = te-tb
#
#    matrix_timing = Timing('matrix', t_py, t_f90)
#    # ...
#
#    # ...
#    print_timing([matrix_timing])
##    print_timing([matrix_timing, rhs_timing, l2norm_timing])
#    # ...



#==============================================================================
def print_timing(ls, headers):
    # ...
    table   = []

    for timing in ls:
        line   = [timing.kind] + timing.time
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
    # ...

#==============================================================================
def run_poisson_1(domain, solution, f, ncells, degree, backend, nprocs=None):

    # ... abstract model
    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of_space(V, 'F')

    v = element_of_space(V, 'v')
    u = element_of_space(V, 'u')

    a = BilinearForm((v,u), dot(grad(v), grad(u)))
    l = LinearForm(v, f*v)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')
    # ...
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells,comm=mpi_comm, nprocs=nprocs)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...
    # dict to store timings
    d = {}

    # ... bilinear form
    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)

    ah.assemble();
    tb = time.time(); M = ah.assemble(); te = time.time()

    d['matrix'] = te-tb
    # ...

    # ... linear form
    lh = discretize(l, domain_h, Vh, backend=backend)

    lh.assemble();
    tb = time.time(); L = lh.assemble(); te = time.time()

    d['rhs'] = te-tb
    # ...

    # ... norm
    # coeff of phi are 0
    phi = FemField( Vh )

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)

    l2norm_h.assemble(F=phi);
    tb = time.time(); err = l2norm_h.assemble(F=phi); te = time.time()


    d['l2norm'] = te-tb
    # ...

    return d

#==============================================================================
def run_poisson_2(domain, solution, f, ncells, degree, backend):

    # ... abstract model
    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of_space(V, 'F')

    v = element_of_space(V, 'v')
    u = element_of_space(V, 'u')

    a = BilinearForm((v,u), dot(grad(v), grad(u)))
    l = LinearForm(v, f*v)


    # ...
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells,comm=mpi_comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...
    # dict to store timings
    d = {}

    # ... bilinear form
    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)

    M = ah.assemble()
    # ...

    # ... linear form
    lh = discretize(l, domain_h, Vh, backend=backend)

    L = lh.assemble()

    M.dot(L);
    tb = time.time(); M.dot(L); te = time.time()
    # ...

    d['matrix_vector'] = te - tb

    L.dot(L);
    tb = time.time(); L.dot(L); te = time.time()
    
    d['vector_vector'] = te - tb
    
    return d

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
def test_perf_poisson_2d_1():
    domain = Square()
    x,y = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    d_f90 = {}
    # using Pyccel
    for n in range(8,9):
        d_d = {}
        for d in range(2, 8):
            d_d[d] = run_poisson_1( domain, solution, f,
                         ncells=[2**n,2**n], degree=[d,d],
                         backend=PSYDAC_BACKEND_GPYCCEL )
        d_f90[n] = d_d

    if mpi_rank == 0:
        keys = sorted(list(d_f90[n][2].keys()))
        timings = []
        for key in keys:
            args = [d[key] for d in d_f90[n].values()]
            timing = Timing(key, args)
            timings += [timing]
        headers = ['Assembly time'] + list(d_f90[n].keys())
        print_timing(timings, headers)
    # ...
def test_perf_poisson_2d_2():
    domain = Square()
    x,y = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    d_f90 = {}
    # using Pyccel
    p1,p2 = int(np.sqrt(mpi_size)),int(np.sqrt(mpi_size))

    for d in range(2,8):
        d_f90[d] = run_poisson_1( domain, solution, f,
                         ncells=[2**3*p1, 2**3*p2], degree=[d, d],
                         backend=PSYDAC_BACKEND_GPYCCEL,nprocs=[p1,p2] )
    

    if mpi_rank == 0:
        keys = d_f90[2].keys()
        timings = []
        for key in keys:
            args = [d_f90[i][key] for i in d_f90.keys()]
            timing = Timing(key, args)
            timings += [timing]

        headers = ['Assembly time'] + list(d_f90.keys())
        print_timing(timings, headers)

def test_perf_poisson_2d_3():
    domain = Square()
    x,y = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    d_f90 = {}
    # using Pyccel

    for d in range(2,8):
        d_f90[d] = run_poisson_2( domain, solution, f,
                         ncells=[2**8, 2**8], degree=[d, d],
                         backend=PSYDAC_BACKEND_GPYCCEL )

    if mpi_rank == 0:
        keys = d_f90[2].keys()
        timings = []
        for key in keys:
            args = [d_f90[i][key] for i in d_f90.keys()]
            timing = Timing(key, args)
            timings += [timing]

        headers = ['Dot_Product'] + list(d_f90.keys())
        print_timing(timings, headers)
###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_perf_poisson_2d_1()
#    test_perf_poisson_2d_2()
#    test_perf_poisson_2d_3()
