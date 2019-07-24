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
from psydac.linalg.utilities import array_to_stencil

from sympy import Tuple
from numpy import linspace, zeros, allclose
from mpi4py import MPI

import pytest
import numpy as np
from scipy.sparse.linalg import spsolve,splu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

kwargs = {'settings':{'solver':'cg', 'tol':1e-9, 'maxiter':10000, 'verbose':False}}
#==============================================================================
def run_laplace_2d_nitsche_dir(solution, f, ncells, degree, s, kappa, comm=None):

    # ... abstract model
    domain = Square()

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
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    # ... solve the equation
    x = equation_h.solve(**kwargs)
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    #k1, k2 = np.pi/3,5*np.pi/3
    #xc, yc = 1/3, 1/4
    #
    #  model = lambda x,y:np.sin(k1*(x-xc))*np.sin(k2*(y-yc))
    #  integrand = lambda *x:(phi(*x)-model(*x))**2
    #
    #  l2_error = np.sqrt( Vh.integral( integrand ) )
    
    # ... compute norms

    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    
   
    return l2_error, h1_error

#==============================================================================
def run_maxwell_2d_nitsche_dir(solution, f, ncells, degree, s, kappa, mu, comm=None):
  
    # ... abstract model
    domain = Square()

    B = domain.boundary

    V = VectorFunctionSpace('V', domain,kind='H1')
    
    nn = NormalVector('nn')

    F = element_of_space(V, name='F')
    
    error    = Tuple(F[0]-solution[0], F[1]-solution[1])
    l2norm  = Norm(error, domain, kind='l2')
    h1norm  = Norm(error, domain, kind='h1')

    E,P = [element_of_space(V, name=i) for i in ['E', 'P']]

    a0  = BilinearForm((E,P), curl(E)*curl(P) + mu*dot(E,P))

    a_B = BilinearForm((E,P), -s*trace_0(cross(P,nn), B)*trace_0(curl(E), B) \
                              -trace_0(cross(E,nn), B)*trace_0(curl(P), B) \
                              +kappa*trace_0(cross(E,nn), B) * trace_0(cross(P,nn), B))
                              
    l0   = LinearForm(P, dot(f,P))
    l_B  = LinearForm(P, -cross(solution,nn)*trace_0(curl(P),B) + kappa*cross(solution,nn)*trace_0(cross(P,nn),B))
    # ...
    
    l   = LinearForm(P, l0(P) + l_B(P))
    a   = BilinearForm((E,P), a0(E,P) + a_B(E,P))

    equation = find(E, forall=P, lhs=a(E,P), rhs=l(P))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...


    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    
    # ... solve the equation
#    x = equation_h.solve(**kwargs)
    equation_h.assemble()
    lhs = equation_h.linear_system.lhs.tosparse().tocsr()
    rhs = equation_h.linear_system.rhs.toarray()
    # ...
    # ...
    x = spsolve(lhs, rhs)
    
    x = array_to_stencil(x, Vh.vector_space)
    
    phi = FemField( Vh, x )
    # ...

    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    
    return l2_error, h1_error



  
def test_api_laplace_2d_nitsche_dir():

    from sympy.abc import x,y

    k1, k2 = pi/3,5*pi/3
    xc, yc = 1/3, 1/4
    
    solution = sin(k1*(x-xc))*sin(k2*(y-yc))
    f        = -solution.diff(x,2) - solution.diff(y,2)
    
    l2_error, h1_error = run_laplace_2d_nitsche_dir(solution, f, ncells=[2**3,2**3], 
                                                    degree=[2,2], s=1, kappa=10**20)
    
    expected_l2_error =  0.00037843634364452175
    expected_h1_error =  0.02192134807021331

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

def test_api_maxwell_2d_nitsche_dir():

    from sympy.abc import x,y


    k1 = 2.0 * pi
    k2 = k1
    xc = 1.0 / 3.0
    yc = 1.0 / 4.0
    mu = 1.0
    
    solution = Tuple(cos(k1*(x-xc))*sin(k2*(y-yc)), 
                    sin(k1*(x-xc))*cos(k2*(y-yc)))

    f        = solution
    
    l2_error, h1_error = run_maxwell_2d_nitsche_dir(solution, f, ncells=[2**5,2**5], 
                                                    degree=[2,2], s=1, kappa=10**20, mu=mu)

    expected_l2_error =  0.0003460526398200829
    expected_h1_error =  0.034239942968633204

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

