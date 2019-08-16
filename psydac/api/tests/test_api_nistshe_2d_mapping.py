# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy.utilities.lambdify import implemented_function

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic   import FemField
from psydac.api.discretization import discretize
from psydac.api.settings import *
from numpy import linspace, zeros, allclose
from mpi4py import MPI
import pytest
import numpy as np
from scipy.sparse.linalg import spsolve,splu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import os

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_laplace_2d_nitsche_dir(filename, solution, f, s, kappa, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    B = domain.boundary

    x,y = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    a0  = BilinearForm((u,v), dot(grad(v),grad(u)))

    a_B = BilinearForm((u,v), -s*trace_0(u, B)*trace_1(grad(v), B) \
                              -trace_0(v, B)*trace_1(grad(u), B) \
                              +kappa*trace_0(u, B) * trace_0(v, B))

    a = BilinearForm((u,v), a0(u,v) + a_B(u,v))

    l0  = LinearForm(v, f*v)
    l_B = LinearForm(v, -s*solution*trace_1(grad(v),B)+ kappa*solution*trace_0(v, B))

    l   = LinearForm(v, l0(v) + l_B(v))

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')


    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    # ... solve the equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)


    return l2_error, h1_error

#==============================================================================
def test_api_laplace_2d_nitsche_dir_1():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    #filename = os.path.join(mesh_dir, 'collela_2d.h5')
    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution


    l2_error, h1_error = run_laplace_2d_nitsche_dir(filename, solution, f, s=1, kappa=10**20)


    expected_l2_error =  0.00015446936830728803
    expected_h1_error =  0.009276141176125671

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_2d_nitsche_dir_2():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution


    l2_error, h1_error = run_laplace_2d_nitsche_dir(filename, solution, f, s=1, kappa=10**10)

    expected_l2_error =  0.01660877902787216
    expected_h1_error =  0.23425587117811927

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_2d_nitsche_dir_3():

    filename = os.path.join(mesh_dir, 'quart_circle.h5')

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y) + x**2*y**2
    f        = (5./4.)*pi**2*cos(0.5*pi*x)*sin(pi*y) - 2*y**2 - 2*x**2


    l2_error, h1_error = run_laplace_2d_nitsche_dir(filename, solution, f, s=1, kappa=10**10)

    expected_l2_error =  0.00015095253299238613
    expected_h1_error =  0.005572269506680323

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
## TODO debug
#def test_api_laplace_2d_nitsche_dir_4():
#
#    filename = os.path.join(mesh_dir, 'annulus.h5')
#
#    from sympy.abc import x,y
#
#    solution = cos(0.5*pi*x)*sin(pi*y) + x**2*y**2
#    f        = (5./4.)*pi**2*cos(0.5*pi*x)*sin(pi*y) - 2*y**2 - 2*x**2
#
#
#    l2_error, h1_error = run_laplace_2d_nitsche_dir(filename, solution, f, s=1, kappa=10**5)
#
#    expected_l2_error =  0.08030849457794145
#    expected_h1_error =  0.5641745896615159
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

