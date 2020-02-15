from collections import OrderedDict
from itertools import product,groupby

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple
from sympy import AtomicExpr
from sympy import Symbol, Mul

from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.topology import IndexedTestTrial
from sympde.topology import ScalarField, VectorField
from sympde.topology import IndexedVectorField
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import Mapping
from sympde.topology import SymbolicDeterminant
from sympde.topology import SymbolicInverseDeterminant
from sympde.topology import SymbolicWeightedVolume
from sympde.topology import IdentityMapping
from sympde.topology import element_of, VectorFunctionSpace, ScalarFunctionSpace
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from .utilities import physical2logical

from pyccel.ast           import AugAssign, Assign
from pyccel.ast.core      import _atomic

from sympde.topology.derivatives import get_index_logical_derivatives, get_atom_logical_derivatives

#==============================================================================
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
class ArityType(with_metaclass(Singleton, Basic)):
    """Base class representing a form type: bilinear/linear/functional"""
    pass

class BilinearArity(ArityType):
    pass

class LinearArity(ArityType):
    pass

class FunctionalArity(ArityType):
    pass

#==============================================================================
class LengthNode(Basic):
    """Base class representing one length of an iterator"""
    pass

class LengthElement(LengthNode):
    pass

class LengthQuadrature(LengthNode):
    pass

class LengthDof(LengthNode):
    pass

class LengthDofTrial(LengthNode):
    pass

class LengthDofTest(LengthNode):
    pass
#==============================================================================
class IndexNode(Basic):
    """Base class representing one index of an iterator"""
    def __new__(cls, index_length):
        assert isinstance(index_length, LengthNode)
        return Basic.__new__(cls, index_length)

    @property
    def length(self):
        return self._args[0]

class IndexElement(IndexNode):
    pass

class IndexQuadrature(IndexNode):
    pass

class IndexDof(IndexNode):
    pass

class IndexDofTrial(IndexNode):
    pass

class IndexDofTest(IndexNode):
    pass

class IndexDerivative(IndexNode):
    def __new__(cls):
        return Basic.__new__(cls)

index_element   = IndexElement(LengthElement())
index_quad      = IndexQuadrature(LengthQuadrature())
index_dof       = IndexDof(LengthDof())
index_dof_trial = IndexDofTrial(LengthDofTrial())
index_dof_test  = IndexDofTest(LengthDofTest())
index_deriv     = IndexDerivative()
#==============================================================================
class RankNode(with_metaclass(Singleton, Basic)):
    """Base class representing a rank of an iterator"""
    pass

class RankDimension(RankNode):
    pass

rank_dim = RankDimension()

#==============================================================================
class BaseNode(Basic):
    """
    """
    pass

#==============================================================================
class Element(BaseNode):
    """
    """
    pass

#==============================================================================
class Pattern(Tuple):
    """
    """
    pass
#==============================================================================
class EvalField(BaseNode):
    def __new__(cls, atoms, index, basis, coeffs, tests, nderiv):

        stmts  = []
        inits  = []
        for v in tests:
            stmts += construct_logical_expressions(v, nderiv)
        atoms   = [physical2logical(i) for i in atoms]

        for coeff in coeffs:
            for a in atoms:
                node    = AtomicNode(a)
                val     = ProductGenerator(MatrixQuadrature(a), index)
                rhs     = Mul(coeff,node)
                stmts  += [AugAssign(val, '+', rhs)]

                inits += [Assign(node,val)]
                inits += [ComputePhysicalBasis(a)]

        inits = Tuple(*inits)
        body  = Loop( basis, index, stmts)

        return Basic.__new__(cls, atoms, inits, body)

    @property
    def atoms(self):
        return self._args[0]

    @property
    def inits(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]
#==============================================================================
class EvalMapping(BaseNode):
    """."""
    def __new__(cls, quads, indices_basis, q_basis, l_basis, mapping, components, space, nderiv):
        atoms  = components.arguments
        basis  = q_basis
        target = basis.target
        if isinstance(target, VectorTestFunction):
            target = target[0]
            #TODO improve how should we do the eval mapping when it's a VectorTestFunction
        new_atoms  = []
        nodes      = []
        l_coeffs   = set()
        for a in atoms:
            atom   = get_atom_logical_derivatives(a)
            node   = a.subs(atom, target)
            new_atoms.append(atom)
            nodes.append(node)
#            if isinstance(node, Matrix):
#                nodes += [*node[:]]
#            else:
#                nodes.append(node)

        stmts = [ComputeLogicalBasis(v,) for v in set(nodes)]
        for i in range(len(atoms)):
            l_coeffs.add(MatrixLocalBasis(new_atoms[i]))
            node    = AtomicNode(nodes[i])
            val     = ProductGenerator(MatrixQuadrature(atoms[i]), quads)
            rhs     = Mul(CoefficientBasis(new_atoms[i]),node)
            stmts  += [AugAssign(val, '+', rhs)]

        loop   = Loop((q_basis, *l_coeffs), quads, stmts)
        loop   = Loop(l_basis, indices_basis, [loop])

        return Basic.__new__(cls, loop, l_coeffs)

    @property
    def loop(self):
        return self._args[0]
    @property
    def coeffs(self):
        return self._args[1]
#==============================================================================
class IteratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies=None):
        if not dummies is None:
            if not isinstance(dummies, (list, tuple, Tuple)):
                dummies = [dummies]
            dummies = Tuple(*dummies)

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorIterator(IteratorBase):
    pass

#==============================================================================
class ProductIterator(IteratorBase):
    pass

#==============================================================================
# TODO dummies should not be None
class GeneratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies):
        if not isinstance(dummies, (list, tuple, Tuple)):
            dummies = [dummies]
        dummies = Tuple(*dummies)

        if not isinstance(target, (ArrayNode, MatrixNode)):
            raise TypeError('expecting an ArrayNode')

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorGenerator(GeneratorBase):
    pass

#==============================================================================
class ProductGenerator(GeneratorBase):
    pass

#==============================================================================
class Grid(BaseNode):
    """
    """
    pass

#==============================================================================
class ScalarNode(BaseNode, AtomicExpr):
    """
    """
    pass

#==============================================================================
class ArrayNode(BaseNode, AtomicExpr):
    """
    """
    _rank = None
    _positions = None
    _free_indices = None

    @property
    def rank(self):
        return self._rank

    @property
    def positions(self):
        return self._positions

    @property
    def free_indices(self):
        if self._free_indices is None:
            return list(self.positions.keys())

        else:
            return self._free_indices

    def pattern(self):
        positions = {}
        for a in self.free_indices:
            positions[a] = self.positions[a]

        args = [None]*self.rank
        for k,v in positions.items():
            args[v] = k

        return Pattern(*args)

#==============================================================================
class MatrixNode(ArrayNode):
    pass

class BlockMatrixNode(MatrixNode):
    pass
#==============================================================================
class GlobalTensorQuadrature(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_indices = [index_element]

#==============================================================================
class LocalTensorQuadrature(ArrayNode):
    # TODO add set_positions
    """
    """
    _rank = 1
    _positions = {index_quad: 0}

#==============================================================================
class TensorQuadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class MatrixQuadrature(MatrixNode):
    """
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class WeightedVolumeQuadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class GlobalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 4
    _positions = {index_quad: 3, index_deriv: 2, index_dof: 1, index_element: 0}
    _free_indices = [index_element]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction, IndexedTestTrial)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedTestTrial):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarTestFunction, IndexedTestTrial))

#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 3
    _positions = {index_quad: 2, index_deriv: 1, index_dof: 0}
    _free_indices = [index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction, IndexedTestTrial)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedTestTrial):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarTestFunction, IndexedTestTrial))
#==============================================================================
class TensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_quad: 1, index_deriv: 0}
    _free_indices = [index_quad]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction, IndexedTestTrial)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedTestTrial):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarTestFunction, IndexedTestTrial))
#==============================================================================
class CoefficientBasis(ScalarNode):
    """
    """
    def __new__(cls, target):
        ls = target.atoms(ScalarTestFunction, VectorTestFunction, Mapping)
        if not len(ls) == 1:
            raise TypeError('Expecting a scalar/vector test function or a Mapping')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]
#==============================================================================
class TensorBasis(CoefficientBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTestBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_test: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTestBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_test: 0}
    _free_indices = [index_dof_test]

#==============================================================================
class TensorQuadratureTestBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTestBasis(TensorBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTrialBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_trial: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTrialBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_trial: 0}
    _free_indices = [index_dof_trial]

#==============================================================================
class TensorQuadratureTrialBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTrialBasis(TensorBasis):
    pass

#==============================================================================
class MatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class StencilMatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

#==============================================================================
class StencilMatrixGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

#==============================================================================
class StencilVectorLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]



#==============================================================================
class StencilVectorGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]



class LocalElementBasis(MatrixNode):
    tag  = random_string( 6 )

class GlobalElementBasis(MatrixNode):
    tag  = random_string( 6 )

class BlockStencilMatrixLocalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self.trials.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class BlockStencilMatrixGlobalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self.trials.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond


class BlockStencilVectorLocalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls,tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self._tests.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class BlockStencilVectorGlobalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self._tests.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond
#==============================================================================
class GlobalSpan(ArrayNode):
    """
    """
    _rank = 1
    _positions = {index_element: 0}

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction, IndexedTestTrial)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Span(ScalarNode):
    """
    """
    def __new__(cls, target=None):
        if not( target is None ):
            if not isinstance(target, (ScalarTestFunction, VectorTestFunction, IndexedTestTrial)):
                raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Evaluation(BaseNode):
    """
    """
    pass

#==============================================================================
class FieldEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class MappingEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class ComputeNode(Basic):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ComputePhysical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputePhysicalBasis(ComputePhysical):
    """
    """
    pass

#==============================================================================
class ComputeKernelExpr(ComputeNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ComputeLogical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputeLogicalBasis(ComputeLogical):
    """
    """
    pass

#==============================================================================
class Reduction(Basic):
    """
    """
    def __new__(cls, op, expr, lhs=None):
        # TODO add verification on op = '-', '+', '*', '/'
        return Basic.__new__(cls, op, expr, lhs)

    @property
    def op(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

#==============================================================================
class Reduce(Basic):
    """
    """
    def __new__(cls, op, rhs, lhs, loop):
        # TODO add verification on op = '-', '+', '*', '/'
        if not isinstance(loop, Loop):
            raise TypeError('Expecting a Loop')

        return Basic.__new__(cls, op, rhs, lhs, loop)

    @property
    def op(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

    @property
    def loop(self):
        return self._args[3]

#==============================================================================
class Reset(Basic):
    """
    """
    def __new__(cls, var, expr=None):
        return Basic.__new__(cls, var, expr)

    @property
    def var(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class ElementOf(Basic):
    """
    """
    def __new__(cls, target):
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class ExprNode(Basic):
    """
    """
    pass

#==============================================================================
class AtomicNode(ExprNode, AtomicExpr):
    """
    """

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ValueNode(ExprNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class PhysicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class LogicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalBasisValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalBasisValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalGeometryValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalGeometryValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        types = (IndexedTestTrial, VectorTestFunction,
                 ScalarTestFunction, IndexedVectorField,
                 ScalarField, VectorField)

        ls = _atomic(expr, cls=types)
        if not(len(ls) == 1):
            raise ValueError('Expecting an expression with one test function')

        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom

#==============================================================================
class GeometryAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        ls = list(expr.atoms(Mapping))
        if not(len(ls) == 1):
            raise ValueError('Expecting an expression with one mapping')

        # TODO
        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom

#==============================================================================
class GeometryExpr(Basic):
    """
    """
    def __new__(cls, expr):
        # TODO assert on expr
        atom = GeometryAtom(expr)
        expr = MatrixQuadrature(expr)

        return Basic.__new__(cls, atom, expr)

    @property
    def atom(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class Loop(BaseNode):
    """
    class to describe a loop of an iterator over a generator.
    """

    def __new__(cls, iterable, index, stmts=None):
        # ...
        if not( isinstance(iterable, (list, tuple, Tuple)) ):
            iterable = [iterable]

        iterable = Tuple(*iterable)
        # ...

        # ... replace GeometryExpressions by a list of expressions
        others = [i for i in iterable if not isinstance(i, GeometryExpressions)]
        geos   = [i.expressions for i in iterable if isinstance(i, GeometryExpressions)]

        if len(geos) == 1:
            geos = list(geos[0])

        elif len(geos) > 1:
            raise NotImplementedError('TODO')

        iterable = others + geos
        iterable = Tuple(*iterable)
        # ...

        # ...
        if not( isinstance(index, IndexNode) ):
            raise TypeError('Expecting an index node')
        # ...

        # ... TODO - add assert w.r.t index type
        #          - this should be splitted/moved somewhere
        iterator = []
        generator = []
        for a in iterable:
            i,g = construct_itergener(a, index)
            iterator.append(i)
            generator.append(g)
        # ...
        # ...
        iterator = Tuple(*iterator)
        generator = Tuple(*generator)
        # ...

        # ...
        if stmts is None:
            stmts = []

        elif not isinstance(stmts, (tuple, list, Tuple)):
            stmts = [stmts]

        stmts = Tuple(*stmts)
        # ...

        obj = Basic.__new__(cls, iterable, index, stmts)
        obj._iterator  = iterator
        obj._generator = generator

        return obj

    @property
    def iterable(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    @property
    def stmts(self):
        return self._args[2]

    @property
    def iterator(self):
        return self._iterator

    @property
    def generator(self):
        return self._generator

    def get_geometry_stmts(self, mapping):
        args = []

        l_quad = list(self.generator.atoms(LocalTensorQuadrature))
        if len(l_quad) == 0:
            return Tuple(*args)

        assert(len(l_quad) == 1)
        l_quad = l_quad[0]

        if isinstance(mapping, IdentityMapping):
            args += [ComputeLogical(WeightedVolumeQuadrature(l_quad))]
            return Tuple(*args)

        args += [ComputeLogical(WeightedVolumeQuadrature(l_quad))]

        # add stmts related to the geometry
        # TODO add other expressions
        args += [ComputeLogical(SymbolicDeterminant(mapping))]
        args += [ComputeLogical(SymbolicInverseDeterminant(mapping))]
        args += [ComputeLogical(SymbolicWeightedVolume(mapping))]

        return Tuple(*args)

#==============================================================================
class TensorIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, TensorIterator) ):
            raise TypeError('Expecting an TensorIterator')

        if not( isinstance(generator, TensorGenerator) ):
            raise TypeError('Expecting a TensorGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class ProductIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, ProductIterator) ):
            raise TypeError('Expecting an ProductIterator')

        if not( isinstance(generator, ProductGenerator) ):
            raise TypeError('Expecting a ProductGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class SplitArray(BaseNode):
    """
    """
    def __new__(cls, target, positions, lengths):
        if not isinstance(positions, (list, tuple, Tuple)):
            positions = [positions]
        positions = Tuple(*positions)

        if not isinstance(lengths, (list, tuple, Tuple)):
            lengths = [lengths]
        lengths = Tuple(*lengths)

        return Basic.__new__(cls, target, positions, lengths)

    @property
    def target(self):
        return self._args[0]

    @property
    def positions(self):
        return self._args[1]

    @property
    def lengths(self):
        return self._args[2]

#==============================================================================
def construct_logical_expressions(u, nderiv):
    if isinstance(u, IndexedTestTrial):
        dim = u.base.space.ldim
    else:
        dim = u.space.ldim

    ops = [dx1, dx2, dx3][:dim]
    r = range(nderiv+1)
    ranges = [r]*dim
    indices = product(*ranges)

    indices = list(indices)
    indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

    args = []
    u = [u] if isinstance(u, (ScalarTestFunction, IndexedTestTrial)) else [u[i] for i in range(dim)]
    for ijk in indices:
        for atom in u:
            for n,op in zip(ijk, ops):
                for i in range(1, n+1):
                    atom = op(atom)
            args.append(atom)
    return [ComputeLogicalBasis(i) for i in args]


#==============================================================================
class GeometryExpressions(Basic):
    """
    """
    def __new__(cls, M, nderiv):
        dim = M.rdim

        ops = [dx1, dx2, dx3][:dim]
        r = range(nderiv+1)
        ranges = [r]*dim
        indices = product(*ranges)

        indices = list(indices)
        indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

        args = []
        for d in range(dim):
            for ijk in indices:
                atom = M[d]
                for n,op in zip(ijk, ops):
                    for i in range(1, n+1):
                        atom = op(atom)
                args.append(atom)

        expressions = [GeometryExpr(i) for i in args]

        args        = Tuple(*args)
        expressions = Tuple(*expressions)
        return Basic.__new__(cls, args, expressions)

    @property
    def arguments(self):
        return self._args[0]

    @property
    def expressions(self):
        return self._args[1]
#==============================================================================
def construct_itergener(a, index):
    """
    """
    # ... create generator
    if isinstance(a, GlobalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadrature()

    elif isinstance(a, LocalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element   = TensorQuadrature()

    elif isinstance(a, GlobalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureTrialBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureTrialBasis(a.target)

    elif isinstance(a, TensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = TensorTrialBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureTestBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureTestBasis(a.target)

    elif isinstance(a, TensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = TensorTestBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureBasis(a.target)

    elif isinstance(a, TensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorBasis(a.target)

    elif isinstance(a, GlobalSpan):
        generator = TensorGenerator(a, index)
        element   = Span(a.target)

    elif isinstance(a, MatrixLocalBasis):
        generator = ProductGenerator(a, index)
        element   = CoefficientBasis(a.target)

    elif isinstance(a, GeometryExpr):
        generator = ProductGenerator(a.expr, index)
        element   = a.atom

    else:
        raise TypeError('{} not available'.format(type(a)))
    # ...

    # ... create iterator
    if isinstance(element, LocalTensorQuadrature):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadrature):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, Span):
        iterator = TensorIterator(element)

    elif isinstance(element, CoefficientBasis):
        iterator = ProductIterator(element)

    elif isinstance(element, GeometryAtom):
        iterator = ProductIterator(element)

    else:
        raise TypeError('{} not available'.format(type(element)))
    # ...

    return iterator, generator

