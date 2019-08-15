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
from sympde.topology import Domain, Line, Square, Cube, two_patches_Square
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC
from sympde.expr.evaluation import TerminalExpr

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

kwargs = {'settings':{'solver':'cg', 'tol':1e-9, 'maxiter':1000, 'verbose':False}}
#==============================================================================
def run_laplace_2d_nitsche_dir(solution, f, ncells, degree, s, kappa, comm=None):

    # ... abstract model
    domain = Square()

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

    V = VectorFunctionSpace('V', domain,kind='h1')
    
    nn = NormalVector('nn')

    F = element_of(V, name='F')
    
    error    = Tuple(F[0]-solution[0], F[1]-solution[1])
    l2norm  = Norm(error, domain, kind='l2')
    h1norm  = Norm(error, domain, kind='h1')

    E,P = [element_of(V, name=i) for i in ['E', 'P']]

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
    equation_h = discretize(equation, domain_h, [Vh, Vh],symbolic_space=[V,V])
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


#==============================================================================
def run_laplace_2d_2_patchs_nitsche_dir(solution, f, ncells, degree, kappa, comm=None):

    # ... abstract model
    domain  = two_patches_Square()
    domain1 = domain.interior.args[0]
    domain2 = domain.interior.args[1]

    V1 = ScalarFunctionSpace('V1', domain1)
    V2 = ScalarFunctionSpace('V2', domain2)

    B = domain.connectivity
    B1 = list(B._data.values())[0][0]
    B2 = list(B._data.values())[0][-1]
    
    bd1 = Union(*domain.boundary.args[::2]) 
    bd2 = Union(*domain.boundary.args[1::2])

    v1 = element_of(V1, name='v1')
    u1 = element_of(V1, name='u1')

    v2 = element_of(V2, name='v2')
    u2 = element_of(V2, name='u2')

    a0  = BilinearForm(((u1,u2),(v1,v2)), dot(grad(v1),grad(u1)) + dot(grad(v2),grad(u2)))

    a_B = BilinearForm(((u1,u2),(v1,v2)),\
                                -0.5*trace_0(u1, B1) * trace_1(grad(v1), B1)\
                                +0.5*trace_0(u2, B2) * trace_1(grad(v2), B2)\
                                +0.5*trace_0(u2, B) * trace_1(grad(v1), B)\
                                 +0.5*trace_0(u1, B) * trace_1(grad(v2), B)\
                                
                                -0.5*trace_0(v1, B1) * trace_1(grad(u1), B1)\
                                +0.5*trace_0(v2, B2) * trace_1(grad(u2), B2)\
                                +0.5*trace_0(v2, B) * trace_1(grad(u1), B)\
                                +0.5*trace_0(v1, B) * trace_1(grad(u2), B)\
                                
                              +kappa*trace_0(u1, B1) * trace_0(v1, B1)\
                              +kappa*trace_0(u2, B2) * trace_0(v2, B2)\
                              -kappa*trace_0(u1, B) * trace_0(v2, B)\
                              -kappa*trace_0(u2, B) * trace_0(v1, B)\
                                    )
                            
    
    
    a = BilinearForm(((u1,u2),(v1,v2)), 
                                       a0((u1,u2),(v1,v2))\
                                       +a_B((u1,u2),(v1,v2))\
                                      )

    l0  = LinearForm((v1,v2), f*v1+ f*v2)
    
    l   = LinearForm((v1,v2), l0(v1,v2))

    bc1 = EssentialBC(u1, 0, bd1.complement(B1))
    bc2 = EssentialBC(u2, 0, bd2.complement(B2))
   
    bc1._index_component = [0]
    bc1._position        = 0
    bc2._index_component = [1]
    bc2._position        = 0
    
    
    equation = find((u1,u2), forall=(v1,v2), lhs=a((u1,u2),(v1,v2)), rhs=l(v1,v2), bc=[bc1, bc2])
    # ...
    for i in equation.bc:
        i.set_position(0)
    # ... create the computational domain from a topological domain
    domain_h1 = discretize(domain1, ncells=[ncells[0]//2,ncells[1]], comm=comm)
    domain_h2 = discretize(domain2, ncells=[ncells[0]//2,ncells[1]], comm=comm)
    # ...

    # ... discrete spaces
    Vh1 = discretize(V1, domain_h1, degree=degree,xmax=[0.5, 1.])
    Vh2 = discretize(V2, domain_h2, degree=degree,xmin=[0.5, 0.])
    # ...
    S = Vh1*Vh2
    
    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h1, [S, S])
    
    equation_h.assemble()
    
    lhs = equation_h.linear_system.lhs.tosparse().tocsr()
    rhs = equation_h.linear_system.rhs.toarray()

    x = spsolve(lhs, rhs)
    
    x = array_to_stencil(x, S.vector_space)
    # ...

    # ...
    phi1 = FemField( Vh1, x[0] )
    phi2 = FemField( Vh2, x[1] )
    # ...
    
    k1, k2 = np.pi,np.pi
    model = lambda x,y:np.sin(k1*x)*np.sin(k2*y)
    
    # ...
    fig, axs   = plt.subplots(1,2, sharey=True)
    fig2, axs2 = plt.subplots(1,2, sharey=True)

    x1      = list(np.linspace( 0., 0.5, 101 ))
    x2      = list(np.linspace( 0.5, 1., 101 ))
    y       = np.linspace( 0., 1., 101 )

    phi1    = np.array( [[phi1(xi, yj) for xi in x1] for yj in y] )
    phi2    = np.array( [[phi2(xi, yj) for xi in x2] for yj in y] )
    
    X1, Y = np.meshgrid(x1, y)
    X2, Y = np.meshgrid(x2, y)
    
    axs[0].contourf(X1, Y, phi1)   
    cp = axs[1].contourf(X2, Y, phi2)
    
    fig.colorbar(cp)
    
    Z1  = np.array( [[model(xi, yj) for xi in x1] for yj in y] )
    Z2  = np.array( [[model(xi, yj) for xi in x2] for yj in y] )
    
    axs2[0].contourf(X1, Y, Z1)   
    cp2 = axs2[1].contourf(X2, Y, Z2) 
    
    fig2.colorbar(cp2)
    
    plt.show()
    

#==========================================================================================
#==========================================================================================
#==========================================================================================

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
                                                    degree=[2,2], s=1, kappa=10**15, mu=mu)

    expected_l2_error =  0.0003460526398200829
    expected_h1_error =  0.034239942968633204

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

def test_api_2_patchs_2d_nitsche_dir():

    from sympy.abc import x,y
    from sympy import pi,sin
    
    solution = sin(pi*x)*sin(pi*y)
    f       = pi**2*solution
    
    run_laplace_2d_2_patchs_nitsche_dir(solution, f, ncells=[2**3,2**3], 
                                                    degree=[2,2], kappa=10**10)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

teardown_module()
test_api_maxwell_2d_nitsche_dir()
