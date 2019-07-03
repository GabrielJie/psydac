# TODO: - replace call to print_expression by SymbolicExpr (may need to use LogicalExpr)

from collections import OrderedDict
from itertools import groupby
import numpy as np

from sympy import Basic
from sympy import symbols, Symbol, IndexedBase, Indexed, Function
from sympy import Mul, Add, Tuple, Min, Max, Pow
from sympy import Matrix, ImmutableDenseMatrix
from sympy import sqrt as sympy_sqrt
from sympy import S as sympy_S
from sympy import Integer, Float
from sympy.core.relational    import Le, Ge
from sympy.logic.boolalg      import And
from sympy import Mod, Abs
from sympy.core.function import AppliedUndef

from pyccel.ast.core import Variable, IndexedVariable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Slice
from pyccel.ast.core import Range, Product
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import FunctionCall
from pyccel.ast.core import Import
from pyccel.ast import Zeros
from pyccel.ast import Import
from pyccel.ast import DottedName
from pyccel.ast import Nil
from pyccel.ast import Len
from pyccel.ast import If, Is, Return
from pyccel.ast import String, Print, Shape
from pyccel.ast import Comment, NewLine
from pyccel.ast.core      import _atomic
from pyccel.ast.utilities import build_types_decorator

from sympde.core import Cross_3d
from sympde.core import Constant
from sympde.core.math import math_atoms_as_str
from sympde.calculus import grad
from sympde.topology import Mapping
from sympde.topology import ScalarField
from sympde.topology import VectorField, IndexedVectorField
from sympde.topology import Boundary, BoundaryVector, NormalVector, TangentVector
from sympde.topology import Covariant, Contravariant
from sympde.topology import ElementArea
from sympde.topology import LogicalExpr
from sympde.topology import SymbolicExpr
from sympde.topology import UndefinedSpaceType
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives
from sympde.topology.space import FunctionSpace, VectorFunctionSpace
from sympde.topology.space import ProductSpace
from sympde.topology.space import ScalarTestFunction
from sympde.topology.space import VectorTestFunction
from sympde.topology.space import IndexedTestTrial
from sympde.topology.space import Trace
from sympde.topology.derivatives import print_expression
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.expr import BilinearForm, LinearForm, Functional, BasicForm

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from .basic import SplBasic
from .evaluation import EvalQuadratureMapping, EvalQuadratureField, EvalQuadratureVectorField
from .utilities import random_string
from .utilities import build_pythran_types_header, variables
from .utilities import compute_normal_vector, compute_tangent_vector
from .utilities import select_loops, filter_product
from .utilities import rationalize_eval_mapping
from .utilities import compute_atoms_expr_mapping
from .utilities import compute_atoms_expr_vector_field
from .utilities import compute_atoms_expr_field
from .utilities import compute_atoms_expr
from .utilities import is_vector_field, is_field

class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, backend=None):


        obj = SplBasic.__new__(cls, 'dot',name='lo_dot',prefix='lo_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initilize()
        return obj

    @property
    def ndim(self):
        return self._ndim

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend


    def _initilize(self):

        ndim = self.ndim
        nrows           = variables('n1:%s'%(ndim+1),  'int')
        pads            = variables('p1:%s'%(ndim+1),  'int')
        indices1        = variables('ind1:%s'%(ndim+1),'int')
        indices2        = variables('i1:%s'%(ndim+1),  'int')
        nrows_extra     = variables('nrows_extra','int',rank=1,cls=IndexedVariable)

        nrows_arg       = variables('nrows', 'int', rank=1,cls=IndexedVariable)
        pads_arg        = variables('pads', 'int', rank=1,cls=IndexedVariable)

        ex,v            = variables('ex','int'), variables('v','real')
        x, out          = variables('x, out','real',cls=IndexedVariable, rank=ndim)
        mat             = variables('mat','real',cls=IndexedVariable, rank=2*ndim)


        init_body  = [Assign(pads[i],pads_arg[i]) for i in range(ndim)]
        init_body += [Assign(nrows[i],nrows_arg[i]) for i in range(ndim)]


        body = []
        ranges = [Range(2*p+1) for p in pads]
        target = Product(*ranges)


        v1 = x[tuple(i+j for i,j in zip(indices1,indices2))]
        v2 = mat[tuple(i+j for i,j in zip(indices1,pads))+tuple(indices2)]
        v3 = out[tuple(i+j for i,j in zip(indices1,pads))]

        body = [AugAssign(v,'+' ,Mul(v1,v2))]
        for i in range(len(ranges)):
            body = [For([indices2[i]],ranges[i],body)]

        #body = [For(indices2, target, body)]
        body.insert(0,Assign(v, 0.0))
        body.append(Assign(v3,v))
        ranges = [Range(i) for i in nrows]
        
        for i in range(len(ranges)):
            body = [For([indices1[i]],ranges[i],body)]
        #target = Product(*ranges)
        #body = [For(indices1,target,body)]

        for dim in range(ndim):
            body.append(Assign(ex,nrows_extra[dim]))


            v1 = [i+j for i,j in zip(indices1, indices2)]
            v2 = [i+j for i,j in zip(indices1, pads)]
            v1[dim] += nrows[dim]
            v2[dim] += nrows[dim]
            v3 = v2
            v1 = x[tuple(v1)]
            v2 = mat[tuple(v2)+ indices2]
            v3 = out[tuple(v3)]

            rows = list(nrows)
            rows[dim] = ex
            ranges = [2*p+1 for p in pads]
            ranges[dim] -= indices1[dim] + 1
            ranges =[Range(i) for i in ranges]
            #target = Product(*ranges)

            for_body = [AugAssign(v, '+',Mul(v1,v2))]
            for i in range(len(ranges)):
                for_body = [For([indices2[i]],ranges[i],for_body)]

            #for_body = [For(indices2, target, for_body)]
            for_body.insert(0,Assign(v, 0.0))
            for_body.append(Assign(v3,v))

            ranges = [Range(i) for i in rows]
            #target = Product(*ranges)
            for i in range(len(ranges)):
                for_body = [For([indices1[i]],ranges[i],for_body)]

            body += for_body

        
        body = init_body + body
        func_args =  (mat, x, out , nrows_arg , nrows_extra, pads_arg)

        self._imports = [Import('product','itertools')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args), 'external_call':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


class VectorDot(SplBasic):

    def __new__(cls, ndim, backend=None):


        obj = SplBasic.__new__(cls, 'dot', name='v_dot', prefix='v_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initilize()
        return obj

    @property
    def ndim(self):
        return self._ndim

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend

    def _initilize(self):

        ndim = self.ndim

        indices = variables('i1:%s'%(ndim+1),'int')
        shape   = variables('n1:%s'%(ndim+1),'int')
        pads    = variables('p1:%s'%(ndim+1),'int')
        out     = variables('out','real')
        x1,x2   = variables('x1, x2','real',rank=ndim,cls=IndexedVariable)

        shape_arg      = variables('shape', 'int', rank=1,cls=IndexedVariable)
        pads_arg       = variables('pads', 'int', rank=1,cls=IndexedVariable)
	
        init_body  = [Assign(pads[i],pads_arg[i]) for i in range(ndim)]
        init_body += [Assign(shape[i],shape_arg[i]) for i in range(ndim)]

        body = []
        ranges = [Range(p,n-p) for n,p in zip(shape,pads)]
        target = Product(*ranges)


        v1 = x1[indices]
        v2 = x2[indices]
        
        body = [AugAssign(out,'+' ,Mul(v1,v2))]
        for i in range(len(ranges)):
            body = [For([indices[i]],ranges[i],body)]

        #body = [For(indices, target, body)]
        body.insert(0,Assign(out, 0.0))
        body.append(Return(out))

        body = init_body + body
        func_args =  (x1, x2 , pads_arg , shape_arg)

        self._imports = [Import('product','itertools')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args), 'external':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)

