# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy.utilities.lambdify import implemented_function

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of_space
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

    V = FunctionSpace('V', domain)

    B = domain.boundary

    x,y = domain.coordinates

    F = element_of_space(V, name='F')

    v = element_of_space(V, name='v')
    u = element_of_space(V, name='u')

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
    
def test_api_laplace_2d_nitsche_dir():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    #filename = os.path.join(mesh_dir, 'collela_2d.h5')
    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    
    l2_error, h1_error = run_laplace_2d_nitsche_dir(filename, solution, f, s=1, kappa=10**20)
 
    print(l2_error, h1_error)
    expected_l2_error =  0.00037843634364452175
    expected_h1_error =  0.02192134807021331

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

test_api_laplace_2d_nitsche_dir()


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
