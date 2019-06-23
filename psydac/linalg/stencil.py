# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np
from scipy.sparse import coo_matrix
from mpi4py import MPI

from psydac.linalg.basic import VectorSpace, Vector, Matrix
from psydac.ddm.cart     import find_mpi_type, CartDecomposition, CartDataExchanger

__all__ = ['StencilVectorSpace','StencilVector','StencilMatrix']

#===============================================================================
class StencilVectorSpace( VectorSpace ):
    """
    Vector space for n-dimensional stencil format. Two different initializations
    are possible:

    - serial  : StencilVectorSpace( npts, pads, periods, dtype=float )
    - parallel: StencilVectorSpace( cart, dtype=float )

    Parameters
    ----------
    npts : tuple-like (int)
        Number of entries along each direction
        (= global dimensions of vector space).

    pads : tuple-like (int)
        Padding p along each direction (number of diagonals is 2*p+1).

    periods : tuple-like (bool)
        Periodicity along each direction.

    dtype : type
        Type of scalar entries.

    cart : psydac.ddm.cart.CartDecomposition
        Tensor-product grid decomposition according to MPI Cartesian topology.

    """
    def __init__( self, *args, **kwargs ):

        if len(args) == 1 or ('cart' in kwargs):
            self._init_parallel( *args, **kwargs )
        else:
            self._init_serial  ( *args, **kwargs )

    # ...
    def _init_serial( self, npts, pads, periods, dtype=float ):

        assert len(npts) == len(pads) == len(periods)
        self._parallel = False

        # Sequential attributes
        self._starts  = tuple( 0   for n in npts )
        self._ends    = tuple( n-1 for n in npts )
        self._pads    = tuple( pads )
        self._periods = tuple( periods )
        self._dtype   = dtype
        self._ndim    = len( npts )

        # Global dimensions of vector space
        self._npts   = tuple( npts )

    # ...
    def _init_parallel( self, cart, dtype=float ):

        assert isinstance( cart, CartDecomposition )
        self._parallel = True

        # Sequential attributes
        self._starts  = cart.starts
        self._ends    = cart.ends
        self._pads    = cart.pads
        self._periods = cart.periods
        self._dtype   = dtype
        self._ndim    = len(cart.starts)

        # Global dimensions of vector space
        self._npts   = cart.npts

        # Parallel attributes
        self._cart         = cart
        self._mpi_type     = find_mpi_type( dtype )
        self._synchronizer = CartDataExchanger( cart, dtype )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension( self ):
        """ The dimension of a vector space V is the cardinality
            (i.e. the number of vectors) of a basis of V over its base field.
        """
        return np.prod( self._npts )

    # ...
    def zeros( self ):
        """
        Get a copy of the null element of the StencilVectorSpace V.

        Returns
        -------
        null : StencilVector
            A new vector object with all components equal to zero.

        """
        return StencilVector( self )
        
    # ...
    def __eq__(self, V):
    
        if self.parallel and V.parallel:
            cond = self._dtype == V._dtype
            cond = cond and self._cart ==  V._cart
            return cond
            
        elif not self.parallel and not V.parallel:
            cond = self.npts == V.npts
            cond = cond and self.pads == V.pads
            cond = cond and self.periods == V.periods
            cond = cond and self.dtype == V.dtype
            return cond
        else:
            return False

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def parallel( self ):
        return self._parallel

    # ...
    @property
    def cart( self ):
        return self._cart if self._parallel else None

    # ...
    @property
    def npts( self ):
        return self._npts

    # ...
    @property
    def starts( self ):
        return self._starts

    # ...
    @property
    def ends( self ):
        return self._ends

    # ...
    @property
    def pads( self ):
        return self._pads

    # ...
    @property
    def periods( self ):
        return self._periods

    # ...
    @property
    def dtype( self ):
        return self._dtype

    # ...
    @property
    def ndim( self ):
        return self._ndim

#===============================================================================
class StencilVector( Vector ):
    """
    Vector in n-dimensional stencil format.

    Parameters
    ----------
    V : psydac.linalg.stencil.StencilVectorSpace
        Space to which the new vector belongs.

    """
    def __init__( self, V ):

        assert isinstance( V, StencilVectorSpace )

        sizes = [e-s+2*p+1 for s,e,p in zip(V.starts, V.ends, V.pads)]
        self._sizes = tuple(sizes)
        self._ndim = len(V.starts)
        self._data  = np.zeros( sizes, dtype=V.dtype )
        self._space = V

        # TODO: distinguish between different directions
        self._sync  = True

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    def dot( self, v ):

        assert isinstance( v, StencilVector )
        assert v._space is self._space
        res = self._dot(self._data, v._data , self.pads, self._data.shape)
        if self._space.parallel:
            res = self._space.cart.comm_cart.allreduce( res, op=MPI.SUM )

        return res

    #...
    @staticmethod
    def _dot(v1, v2, pads, shape):
        index = tuple( slice(p,n-p) for p,n in zip(pads, shape))
        return np.dot(v1[index].flat, v2[index].flat)

    #...
    def copy( self ):
        w = StencilVector( self._space )
        w._data[:] = self._data[:]
        w._sync    = self._sync
        w._dot     = self._dot
        return w

    #...
    def __mul__( self, a ):
        w = StencilVector( self._space )
        w._data = self._data * a
        w._sync = self._sync
        w._dot  = self._dot
        return w

    #...
    def __rmul__( self, a ):
        w = StencilVector( self._space )
        w._data = a * self._data
        w._sync = self._sync
        w._dot  = self._dot
        return w

    #...
    def __add__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data  +  v._data
        w._sync = self._sync and v._sync
        w._dot  = self._dot
        return w

    #...
    def __sub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data  -  v._data
        w._sync = self._sync and v._sync
        w._dot  = self._dot
        return w

    #...
    def __imul__( self, a ):
        self._data *= a
        return self

    #...
    def __iadd__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data += v._data
        self._sync  = v._sync and self._sync
        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data -= v._data
        self._sync  = v._sync and self._sync
        return self

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def starts(self):
        return self._space.starts

    # ...
    @property
    def ends(self):
        return self._space.ends

    # ...
    @property
    def pads(self):
        return self._space.pads

    # ...
    def __str__(self):
        txt  = '\n'
        txt += '> starts  :: {starts}\n'.format( starts= self.starts )
        txt += '> ends    :: {ends}\n'  .format( ends  = self.ends   )
        txt += '> pads    :: {pads}\n'  .format( pads  = self.pads   )
        txt += '> data    :: {data}\n'  .format( data  = self._data  )
        txt += '> sync    :: {sync}\n'  .format( sync  = self._sync  )
        return txt

    # ...
    def toarray( self, *, with_pads=False ):
        """
        Return a numpy 1D array corresponding to the given StencilVector,
        with or without pads.

        Parameters
        ----------
        with_pads : bool
            If True, include pads in output array.

        Returns
        -------
        array : numpy.ndarray
            A copy of the data array collapsed into one dimension.

        """
        # In parallel case, call different functions based on 'with_pads' flag
        if self.space.parallel:
            if with_pads:
                return self._toarray_parallel_with_pads()
            else:
                return self._toarray_parallel_no_pads()

        # In serial case, ignore 'with_pads' flag
        index = tuple( slice(p,-p) for p in self.pads )
        return self._data[index].flatten()

    # ...
    def _toarray_parallel_no_pads( self ):
        a         = np.zeros( self.space.npts )
        idx_from  = tuple( slice(p,-p) for p in self.pads )
        idx_to    = tuple( slice(s,e+1) for s,e in zip(self.starts,self.ends) )
        a[idx_to] = self._data[idx_from]
        return a.reshape(-1)

    # ...
    def _toarray_parallel_with_pads( self ):

        # Step 0: create extended n-dimensional array with zero values
        shape = tuple( n+2*p for n,p in zip( self.space.npts, self.pads ) )
        a = np.zeros( shape )

        # Step 1: write extended data chunk (local to process) onto array
        idx = tuple( slice(s,e+2*p+1) for s,e,p in
            zip( self.starts, self.ends, self.pads ) )
        a[idx] = self._data

        # Step 2: if necessary, apply periodic boundary conditions to array
        ndim = self.space.ndim

        for direction in range( ndim ):

            periodic = self.space.cart.periods[direction]
            coord    = self.space.cart.coords [direction]
            nproc    = self.space.cart.nprocs [direction]

            if periodic:

                p = self.pads[direction]

                # Left-most process: copy data from left to right
                if coord == 0:
                    idx_from = tuple(
                        (slice(None,p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    idx_to = tuple(
                        (slice(-2*p,-p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    a[idx_to] = a[idx_from]

                # Right-most process: copy data from right to left
                if coord == nproc-1:
                    idx_from = tuple(
                        (slice(-p,None) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    idx_to = tuple(
                        (slice(p,2*p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    a[idx_to] = a[idx_from]

        # Step 3: remove ghost regions from global array
        idx = tuple( slice(p,-p) for p in self.pads )
        out = a[idx]

        # Step 4: return flattened array
#        return out.flatten()
        return out.reshape(-1)

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

    # ...
    @property
    def ghost_regions_in_sync( self ):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value

    # ...
    # TODO: maybe change name to 'exchange'
    def update_ghost_regions( self, *, direction=None ):
        """
        Update ghost regions before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        if self.space.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self.space._synchronizer.update_ghost_regions( self._data, direction=direction )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial( direction )

        # Flag ghost regions as up-to-date
        self._sync = True

    # ...
    def _update_ghost_regions_serial( self, direction=None ):

        if direction is None:
            for d in range( self._space.ndim ):
                self._update_ghost_regions_serial( d )
            return

        ndim     = self._space.ndim
        periodic = self._space.periods[direction]
        p        = self._space.pads   [direction]

        idx_front = [slice(None)]*direction
        idx_back  = [slice(None)]*(ndim-direction-1)

        if periodic:

            # Copy data from left to right
            idx_from = tuple( idx_front + [slice( p, 2*p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

            # Copy data from right to left
            idx_from = tuple( idx_front + [slice(-2*p,-p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

        else:

            # Set left ghost region to zero
            idx_ghost = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_ghost] = 0

            # Set right ghost region to zero
            idx_ghost = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_ghost] = 0

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        # TODO: check if we should ignore padding elements

        if not isinstance( key, tuple ):
            key = (key,)
        index = []
        for (i,s,p) in zip(key, self.starts, self.pads):
            if isinstance(i, slice):
                start = None if i.start is None else i.start - s + p
                stop  = None if i.stop  is None else i.stop  - s + p
                l = slice(start, stop, i.step)
            else:
                l = i - s + p
            index.append(l)
        return tuple(index)

#===============================================================================
class StencilMatrix( Matrix ):
    """
    Matrix in n-dimensional stencil format.

    This is a linear operator that maps elements of stencil vector space V to
    elements of stencil vector space W.

    For now we only accept V==W.

    Parameters
    ----------
    V : psydac.linalg.stencil.StencilVectorSpace
        Domain of the new linear operator.

    W : psydac.linalg.stencil.StencilVectorSpace
        Codomain of the new linear operator.

    """
    def __init__( self, V, W, pads=None ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert W.pads == V.pads
        
        self._pads     = pads or tuple(V.pads)
        dims           = [e-s+2*p+1 for s,e,p in zip(W.starts, W.ends, W.pads)]
        diags          = [2*p+1 for p in self._pads]
        self._data     = np.zeros( dims+diags, dtype=W.dtype )
        self._domain   = V
        self._codomain = W
        self._ndim     = len( dims )

        # Parallel attributes
        if V.parallel:
            # Create data exchanger for ghost regions
            self._synchronizer = CartDataExchanger(
                cart        = V.cart,
                dtype       = V.dtype,
                coeff_shape = diags
            )

        # Flag ghost regions as not up-to-date (conservative choice)
        self._sync = False

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._domain

    # ...
    @property
    def codomain( self ):
        return self._codomain

    # ...
    def dot( self, v, out=None ):

        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        # Necessary if vector space is distributed across processes
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        # Shortcuts
        ssc = self.codomain.starts
        eec = self.codomain.ends
        ssd = self.domain.starts
        eed = self.domain.ends
        pp = self.pads

        # Number of rows in matrix (along each dimension)
        nrows       = [ed-s+1 for s,ed in zip(ssd, eed)]
        nrows_extra = [0 if ec<=ed else ec-ed for ec,ed in zip(eec,eed)]

        self._dot(self._data, v._data, out._data, nrows, nrows_extra, pp)

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    @staticmethod
    def _dot(mat, x, out, nrows, nrows_extra, pads):

        # Index for k=i-j
        ndim = len(x.shape)
        kk = [slice(None)]*ndim

        for xx in np.ndindex( *nrows ):

            ii    = tuple( p+x for p,x in zip(pads,xx) )
            jj    = tuple( slice(x,x+2*p+1) for x,p in zip(xx,pads) )
            ii_kk = tuple( list(ii) + kk )

            out[ii] = np.dot( mat[ii_kk].flat, x[jj].flat )

        for d,er in enumerate(nrows_extra):
        
            ee = [0]*len(nrows_extra)
            kk = [slice(None)] *ndim
            rows = nrows.copy()
            del rows[d]
            
            
            for n in range(er):
                ee[d] = n+1
                for xx in np.ndindex(*rows):
                    ii = [*xx]
                    ii.insert(d, nrows[d]+n)
        
                    ii    = tuple(i+p for i,p in zip(ii, pads))
                    jj    = tuple( slice(i, i+2*p+1-e) for i,p,e in zip(ii, pp, ee) )

                    kk[d] = slice(None,2*pp[d]-n)
                    
                    ii_kk = tuple( list(ii) + kk )

                    v1 = x[jj]
                    v2 = mat[ii_kk]
                    out[ii] = np.dot( v2.flat, v1.flat )

    # ...
    def toarray( self, *, with_pads=False ):

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads()
        else:
            coo = self._tocoo_no_pads()

        return coo.toarray()

    # ...
    def tosparse( self, *, with_pads=False ):

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads()
        else:
            coo = self._tocoo_no_pads()

        return coo

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    # ...
    @property
    def pads( self ):
        return self._pads

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value


    #...
    def max( self ):
        return self._data.max()

    #...
    def copy( self ):
        M = StencilMatrix( self.domain, self.codomain, self._pads )
        M._data[:] = self._data[:]
        M._dot = self._dot
        return M

    #...
    def __mul__( self, a ):
        w = StencilMatrix( self._domain, self._codomain, self._pads )
        w._data = self._data * a
        w._sync = self._sync
        return w

    #...
    def __rmul__( self, a ):
        w = StencilMatrix( self._domain, self._codomain, self._pads )
        w._data = a * self._data
        w._sync = self._sync
        return w

    # ...
    def __neg__(self):
        return self.__mul__(-1)


    #...
    def remove_spurious_entries( self ):
        """
        If any dimension is NOT periodic, make sure that the corresponding
        periodic corners are set to zero.

        """
        # TODO: access 'self._data' directly for increased efficiency
        # TODO: add unit tests

        ndim  = self._domain.ndim

        for direction in range(ndim):

            periodic = self._domain.periods[direction]

            if not periodic:

                nc = self._codomain.npts[direction]
                nd = self._domain.npts[direction]

                s = self._codomain.starts[direction]
                e = self._codomain.ends  [direction]
                p = self.pads  [direction]

                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Top-right corner
                for i in range( max(0,s), min(p,e+1) ):
                    index = tuple( idx_front + [i]            + idx_back +
                                   idx_front + [slice(-p,-i)] + idx_back )
                    self[index] = 0

                # Bottom-left corner
                for i in range( max(nd-p,s), min(nc,e+1) ):
                    index = tuple( idx_front + [i]              + idx_back +
                                   idx_front + [slice(nd-i,p+1)] + idx_back )
                    self[index] = 0

    # ...
    def update_ghost_regions( self, *, direction=None ):
        """
        Update ghost regions before performing non-local access to matrix
        elements (e.g. in matrix transposition).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.update_ghost_regions( self._data, direction=direction )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial( direction )

        # Flag ghost regions as up-to-date
        self._sync = True

    # ...
    def transpose( self ):
        """ Create new StencilMatrix Mt, where domain and codomain are swapped
            with respect to original matrix M, and Mt_{ij} = M_{ji}.
        """
        # For clarity rename self
        M = self

        # If necessary, update ghost regions of original matrix M
        if not M.ghost_regions_in_sync:
            M.update_ghost_regions()

        # Create new matrix where domain and codomain are swapped
        Mt = StencilMatrix(M.codomain, M.domain, pads=self._pads)

        # Number of rows in matrix (along each dimension)
        nrows = [e - s + 1 for s, e in zip(M._codomain.starts, M._codomain.ends)]
        nrows_extra = []

        # Call low-level '_transpose' function (works on Numpy arrays directly)
        self._transpose(M._data, Mt._data, nrows, nrows_extra, M.pads)

        return Mt

    # ...
    @staticmethod
    def _transpose( M, Mt, nrows, nrows_extra, pads ):

        # NOTE:
        #  . Array M  index by [i1, i2, ..., k1, k2, ...]
        #  . Array Mt index by [j1, j2, ..., l1, l2, ...]

        pp = pads
        ndiags = [2*p + 1 for p in pp]

        for xx in np.ndindex( *nrows ):

            jj = tuple(p + x for p, x in zip(pp, xx) )

            for ll in np.ndindex( *ndiags ):

                ii = tuple(  x + l for x, l in zip(xx, ll))
                kk = tuple(2*p - l for p, l in zip(pp, ll))

                Mt[(*jj, *ll)] = M[(*ii, *kk)]

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for i,s,p in zip( ii, self._codomain.starts, self._codomain.pads ):
            x = self._shift_index( i, p-s )
            index.append( x )

        for k,p in zip( kk, self._pads ):
            l = self._shift_index( k, p )
            index.append( l )

        return tuple(index)

    # ...
    @staticmethod
    def _shift_index( index, shift ):
        if isinstance( index, slice ):
            start = None if index.start is None else index.start + shift
            stop  = None if index.stop  is None else index.stop  + shift
            return slice(start, stop, index.step)
        else:
            return index + shift

    #...
    def _tocoo_no_pads( self ):

        # Shortcuts
        nr = self._codomain.npts
        nd = self._ndim
        nc = self._domain.npts
        ss = self._codomain.starts
        pp = self._codomain.pads
        

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )

        for (index,value) in np.ndenumerate( self._data[local] ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # x=i-s
            ll = index[nd:]  # l=p+k

            ii = [s+x for s,x in zip(ss,xx)]
            jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self._pads)]

            I = ravel_multi_index( ii, dims=nr, order='C' )
            J = ravel_multi_index( jj, dims=nc, order='C' )

            rows.append( I )
            cols.append( J )
            data.append( value )

        M = coo_matrix(
                (data,(rows,cols)),
                shape = [np.prod(nr),np.prod(nc)],
                dtype = self._domain.dtype
        )

        M.eliminate_zeros()

        return M

    #...
    def _tocoo_parallel_with_pads( self ):

        # If necessary, update ghost regions
        if not self.ghost_regions_in_sync:
            self.update_ghost_regions()

        # Shortcuts
        nr = self._codomain.npts
        nc = self._domain.npts
        nd = self._ndim

        ss = self._codomain.starts
        ee = self._codomain.ends
        pp = self._pads
        pc = self._codomain.pads
        pd = self._domain.pads
        cc = self._codomain.periods

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        # List of rows (to avoid duplicate updates)
        I_list = []

        # Shape of row and diagonal spaces
        xx_dims = self._data.shape[:nd]
        ll_dims = self._data.shape[nd:]

        # Cycle over rows (x = p + i - s)
        for xx in np.ndindex( *xx_dims ):

            # Compute row multi-index with simple shift
            ii = [s + x - p for (s, x, p) in zip(ss, xx, pc)]

            # Apply periodicity where appropriate
            ii = [i - n if (c and i >= n and i - n < s) else
                  i + n if (c and i <  0 and i + n > e) else i
                  for (i, s, e, n, c) in zip(ii, ss, ee, nr, cc)]

            # Compute row flat index
            # Exclude values outside global limits of matrix
            try:
                I = ravel_multi_index( ii, dims=nr, order='C' )
            except ValueError:
                continue

            # If I is a new row, append it to list of rows
            # DO NOT update same row twice!
            if I not in I_list:
                I_list.append( I )
            else:
                continue

            # Cycle over diagonals (l = p + k)
            for ll in np.ndindex( *ll_dims ):

                # Compute column multi-index (k = j - i)
                jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,pp)]

                # Compute column flat index
                J = ravel_multi_index( jj, dims=nc, order='C' )

                # Extract matrix value
                value = self._data[(*xx, *ll)]

                # Append information to COO arrays
                rows.append( I )
                cols.append( J )
                data.append( value )

        # Create Scipy COO matrix
        M = coo_matrix(
                (data,(rows,cols)),
                shape = [np.prod(nr), np.prod(nc)],
                dtype = self._domain.dtype
        )

        M.eliminate_zeros()

        return M

    # ...
    @property
    def ghost_regions_in_sync( self ):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value

    # ...
    def _update_ghost_regions_serial( self, direction: int ):

        if direction is None:
            for d in range( self._codomain.ndim ):
                self._update_ghost_regions_serial( d )
            return

        ndim     = self._codomain.ndim
        periodic = self._codomain.periods[direction]
        p        = self._codomain.pads   [direction]

        idx_front = [slice(None)]*direction
        idx_back  = [slice(None)]*(ndim-direction-1 + ndim)

        if periodic:

            # Copy data from left to right
            idx_from = tuple( idx_front + [slice( p, 2*p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

            # Copy data from right to left
            idx_from = tuple( idx_front + [slice(-2*p,-p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

        else:

            # Set left ghost region to zero
            idx_ghost = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_ghost] = 0

            # Set right ghost region to zero
            idx_ghost = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_ghost] = 0

#===============================================================================
del VectorSpace, Vector, Matrix
