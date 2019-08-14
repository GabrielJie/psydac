# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call


from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary, Connectivity as sym_Connectivity
from sympde.expr     import Norm as sym_Norm
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Mapping

from psydac.api.basic           import BasicDiscrete
from psydac.api.basic           import random_string
from psydac.api.grid            import QuadratureGrid, BoundaryQuadratureGrid
from psydac.api.grid            import BasisValues
from psydac.api.ast.fem         import Kernel
from psydac.api.ast.fem         import Assembly
from psydac.api.ast.fem         import Interface
from psydac.api.ast.glt         import GltKernel
from psydac.api.ast.glt         import GltInterface
from psydac.api.glt             import DiscreteGltExpr

from psydac.linalg.stencil      import StencilVector, StencilMatrix
from psydac.linalg.block        import BlockVector, BlockLinearOperator
from psydac.cad.geometry        import Geometry
from psydac.mapping.discrete    import SplineMapping, NurbsMapping
from psydac.fem.vector          import ProductFemSpace

import inspect
import sys
import numpy as np


#==============================================================================
class DiscreteBilinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping
        # ...
        self._spaces = args[1]
        # ...
        kwargs['discrete_space']      = self.spaces
        kwargs['mapping']             = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm
        
        boundary = kwargs.pop('boundary', [])
        if boundary and isinstance(boundary, list):
            kwargs['boundary'] = boundary[0]
        elif boundary:
            kwargs['boundary'] = boundary

        test_space  = self.spaces[0]
        trial_space = self.spaces[1]
        
        unique_grid = test_space.unique_grid and trial_space.unique_grid
        kwargs['unique_grid'] = unique_grid
        
        # ...
        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order  = kwargs.pop('quad_order', None)
        boundary    = kwargs.pop('boundary',   None)
        target      = kwargs.pop('target', None)
        # ...

        # ...
        if not unique_grid:
            assert isinstance(test_space, ProductFemSpace)
            test_space  = test_space.spaces
   
            assert isinstance(trial_space, ProductFemSpace)
            trial_space = trial_space.spaces
            if boundary is None:

                for space in test_space:
                    if space.symbolic_space.domain==target:
                        test_space = [space]
                        break
                        
                for space in trial_space:
                    if space.symbolic_space.domain==target:
                        trial_space = [space]
                        break
            elif isinstance(boundary, Boundary):

                for space in test_space:
                    if space.symbolic_space.domain==boundary.domain:
                        test_space = [space]
                        break
                        
                for space in trial_space:
                    if space.symbolic_space.domain==boundary.domain:
                        trial_space = [space]
                        break
            
        else:
            test_space  = [test_space]
            trial_space = [trial_space]
        
        if boundary is None:
            self._grid = [QuadratureGrid( space, quad_order=quad_order ) for space in test_space]

        elif isinstance(boundary, sym_Boundary):
            self._grid = [BoundaryQuadratureGrid( space,
                                                  boundary.axis,
                                                  boundary.ext,
                                                  quad_order=quad_order) for space in test_space]
                                                 
        elif isinstance(boundary, sym_Connectivity):
            #TODO improve
            edge = list(boundary)[0]
            boundary = boundary[edge]
            axis     = boundary[0].axis
            ext      = [bd.ext for bd in boundary]
            
            assert len(test_space) == 2
            self._grid = [BoundaryQuadratureGrid( space, axis, ext, quad_order=quad_order ) 
                          for space,ext in zip(test_space,ext)]


        # ...
        self._test_basis = BasisValues( test_space, self.grid,
                                        nderiv = self.max_nderiv )
        self._trial_basis = BasisValues( trial_space, self.grid,
                                         nderiv = self.max_nderiv )

    @property
    def spaces(self):
        return self._spaces

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def trial_basis(self):
        return self._trial_basis

    def assemble(self, **kwargs):
        newargs = tuple(self.spaces) + (*self.grid, self.test_basis, self.trial_basis)
        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)

#==============================================================================
class DiscreteLinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping
        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping
        # ...

        # ...
        self._space = args[1]
        # ...

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        boundary = kwargs.pop('boundary', [])
        if boundary and isinstance(boundary, list):
            kwargs['boundary'] = boundary[0]
        elif boundary:
            kwargs['boundary'] = boundary

        test_space  = self.space
        unique_grid = test_space.unique_grid
        kwargs['unique_grid'] = unique_grid
        
        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...

        quad_order  = kwargs.pop('quad_order', None)
        boundary    = kwargs.pop('boundary',   None)
        target      = kwargs.pop('target', None)
        

        # ...
        # ...
        if not unique_grid:
            assert isinstance(test_space, ProductFemSpace)
            test_space  = test_space.spaces

            if boundary is None:

                for space in test_space:
                    if space.symbolic_space.domain==target:
                        test_space = [space]
                        break
            elif isinstance(boundary, Boundary):

                for space in test_space:
                    if space.symbolic_space.domain==boundary.domain:
                        test_space = [space]
                        break        
        else:
            test_space  = [test_space]
            trial_space = [trial_space]
            
        # ...
        if boundary is None:
            self._grid = [QuadratureGrid( space, quad_order = quad_order ) for space in test_space] 

        elif isinstance(boundary, sym_Boundary):
            axis = boundary.axis
            ext  = boundary.ext
            self._grid = [BoundaryQuadratureGrid( space, axis, ext,
                         quad_order = quad_order ) for space in test_space]

        elif isinstance(boundary, sym_Connectivity):
                edge = list(boundary)[0]
                boundary = boundary[edge]
                axis     = boundary[0].axis
                ext      = [bd.ext for bd in boundary]
                self._grid = [BoundaryQuadratureGrid( space, axis,
                             e, quad_order = quad_order ) for space,e in zip(test_space, ext)]

        # ...

        self._test_basis = BasisValues( test_space, self.grid,
                                        nderiv = self.max_nderiv )
        # ...

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    def assemble(self, **kwargs):
        newargs = (self.space, *self.grid, self.test_basis)       
        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)
        return self.func(*newargs, **kwargs)


#==============================================================================
class DiscreteFunctional(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_Functional):
            raise TypeError('> Expecting a symbolic Functional')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping
        # ...

        # ...
        self._space = args[1]
        # ...

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order = kwargs.pop('quad_order', None)
        boundary   = kwargs.pop('boundary',   None)
        # ...

        # ...
        if boundary is None:
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )

        else:
            self._grid = BoundaryQuadratureGrid( self.space,
                                                 boundary.axis,
                                                 boundary.ext,
                                                 quad_order = quad_order )
        # ...

        # ...
        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv )
        # ...

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    def assemble(self, **kwargs):
        newargs = (self.space, self.grid, self.test_basis)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        v = self.func(*newargs, **kwargs)

#        # ... TODO remove => this is for debug only
#        import sys
#        sys.path.append(self.folder)
#        from interface_pt3xujb5 import  interface_pt3xujb5
#        sys.path.remove(self.folder)
#        v = interface_pt3xujb5(*newargs, **kwargs)
#        # ...

        # case of a norm
        
        if isinstance(self.expr, sym_Norm):
            if not( self.comm is None ):
                v = self.comm.allreduce(sendobj=v)

            if self.expr.exponent == 2:
                # add abs because of 0 machine
                v = np.sqrt(np.abs(v))
            else:
                raise NotImplementedError('TODO')
        return v


#==============================================================================
class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Functional)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearFormn Functional')

        self._expr = a

        backend = kwargs.get('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string( 8 )

        # ...
        forms = []
        boundaries = kwargs.pop('boundary', [])
        
        unique_grid = True
        for e in kernel_expr:
            if isinstance(e.target, sym_Connectivity):
                unique_grid = False

        for e in kernel_expr:
            kwargs['target'] = e.target
            if isinstance(e.target, (sym_Boundary, sym_Connectivity)):
                boundary = [i for i in boundaries if i is e.target]
                if boundary: kwargs['boundary'] = boundary[0]

            if isinstance(a, sym_BilinearForm):
                ah = DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

            elif isinstance(a, sym_LinearForm):
                ah = DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

            elif isinstance(a, sym_Functional):
                ah = DiscreteFunctional(a, kernel_expr, *args, **kwargs)

            forms.append(ah)
            kwargs['boundary'] = None

        self._forms = forms
        # ...

    @property
    def forms(self):
        return self._forms

    def assemble(self, **kwargs):
        form = self.forms[0]
        M = form.assemble(**kwargs)

        for form in self.forms[1:]:
            
            if isinstance(M, (StencilVector, StencilMatrix)):
                M = {0: M}
            elif isinstance(M, BlockVector):
                M = {i:M[i] for i in range(len(M.blocks))}
            elif isinstance(M, BlockLinearOperator):
                M = {(i,j):M[i,j] for i in range(M.n_block_rows) 
                    for j in range(M.n_block_cols)}

            if M:
                B = form.assemble(**kwargs)
                if isinstance(B, (StencilVector, StencilMatrix)):
                    B = [B]
                for key in M:
                    if M[key] and B[key]:
                        B[key]._data += M[key]._data
                    elif M[key]:
                        assert B[key] is None
                        B[key] = M[key]
                    elif B[key]:
                        assert M[key] is None
                
                if isinstance(B, list):
                    assert len(B) == 1
                    B = B[0]

                M = B
            else:
                M = form.assemble(**kwargs)

        return M
