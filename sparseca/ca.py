import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from datetime import datetime
import logging
from multiprocessing import Pool


def getvectors(X):
    for col in range(X.shape[1]):
        yield X[:, col]


def dotprod(v):
    return v.transpose().dot(v).todense().item()


class SparseCA(object):
    """Perform correspondence analysis on a large matrix.
    
    Positional arguments
    --
    D :: matrix (or sparse matrix) with dimensions (n, m)
    
    Keyword arguments
    --
    cores        :: integer, number of cores to use while calculating inertia 
                    (defaults to 1)
    random_state :: integer or float, the seed for random number generator (set 
                    this for reproducibility)
    k            :: integer, number of singular values (dimensions) to compute 
                    in Truncated SVD
    n_iter       :: integer, number of iterations to perform during 
                    Truncated SVD
    
    Details
    --
    
    Implements correspondence analysis algorithms as described in:
        Greenacre, Michael. Correspondence Analysis in Practice. 2007. 2nd 
            Edition. Boca Raton, Fl.: Chapman and Hall/CRC Press.
    
    The innovation in this implementation is to use Truncated SVD and a few 
    other tricks to reduce the memory complexity of the algorithm. As a result, 
    the algorithm is tractable for larger matrices that would defeat existing 
    implementations on most desktop machines.
    
    This implementation calculates the total inertia of the correspondence 
    matrix by computing the trace of the squared correspondence matrix (C'C), 
    and it computes the trace by summing over the dot products of each column 
    vector with itself. As a result, this implementation is most efficient
    when N >> M, that is, when the number of rows is much greater than the 
    number of columns. 
    
    That final step is the only one currently parallelized using multiprocessing
    (if cores > 1).
    
    """

    def _cor_svd(self, D, k=2, n_iter=7):
        """Generate the correspondence matrix and its SVD from an input matrix D

        inputs
        --
        D :: Matrix (or sparse matrix) with dimensions nxm
        k :: number of singular values
        n_iter :: number of iterations for SVD solver

        outputs
        --
        U, S, V :: Tuple of matrices: nxn matrix U, array S (of eigenvalues), 
        mxm matrix Vt

        Uses the randomized SVD solver from scikit-learn. That implementation's 
        runtime increases more slowly in number of rows than the scipy truncated
        SVD solver.

        See: http://blog.explainmydata.com/2016/01/how-much-faster-is-truncated-svd.html
        And: https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
        """

        n = D.sum()

        logging.info('Computing the matrix P.')

        P = D.multiply(1/n)
        logging.info('The matrix P is a %s x %s matrix with %s nonzero elements.' %
                     (P.shape[0], P.shape[1], P.getnnz()))
        logging.info('Calculating row and column sums.')

        # In general store vectors as column vectors
        r = P.sum(1)                # Sum across the columns to give row sums
        c = P.sum(0).transpose()    # Sum across the rows to give column sums

        logging.info('Computing the matrix of expected proportions.')

        # this will be a large dense matrix
        P_exp = sparse.csc_matrix(r.dot(c.transpose()))  
          
        logging.info('Computing the matrix of deviations (Obs - Exp).')

        P = P - P_exp

        logging.info('Deleting P_exp and D, which are no longer needed.')
        del P_exp
        del D

        logging.info('Computing matrices with row and column sums across the main diagonal.')

        D_r = sparse.diags([r.transpose()], [0], 
                           shape=(r.shape[0], r.shape[0])).tocsr()
        D_c = sparse.diags([c.transpose()], [0], 
                           shape=(c.shape[0], c.shape[0])).tocsr()

        # C :: CORRESPONDENCE MATRIX (of standardized residuals; 
        # Greenacre 2007, p. 242)
        # C and not S because S is the matrix of (S)ingular values in scipy's 
        # SVD implementation

        logging.info('Computing the correspondence matrix C')

        C = D_r.power(-0.5).dot(P).dot(D_c.power(-0.5))

        logging.info('Storing matrix level statistics.')        
        self.nrow, self.ncol = C.shape
        self.nonzero = C.getnnz()

        logging.info('Shape:   %sx%s' % (self.nrow, self.ncol))
        logging.info('Nonzero: %s' % self.nonzero)
        
        
        logging.info('Performing truncated SVD and storing %s singular values.' % str(k))
        if self._seed:            
            U, S, Vt = randomized_svd(C,
                                      n_components=k,
                                      n_iter=n_iter,
                                      flip_sign=True,
                                      random_state=self._seed)
        else:
            U, S, Vt = randomized_svd(C,
                                      n_components=k,
                                      n_iter=n_iter,
                                      flip_sign=True
                                      )

        logging.info('SVD complete.')
        logging.info('Storing (approximate) inertia ...')
        
        # Note that "inertia" here only sums over the calculated values alpha.
        # If the trailing values are substantially greater than zero, this 
        # calculation is inaccurate.
        self.inertia = np.sum(S ** 2)
        logging.info('Inertia: %s' % self.inertia)

        V = Vt.transpose()

        logging.info('SparseCA run is finished.')

        return U, S, V, r, c

    def rsc(self):
        """Get row standard coordinates
        """
        
        D_r = sparse.diags([self.r.transpose()], [0], 
                           shape=(self.r.shape[0], self.r.shape[0])).tocsr()

        return sparse.csc_matrix(D_r.power(-0.5).dot(self.U))

    def csc(self):
        """Get column standard coordinates
        """
        
        D_c = sparse.diags([self.c.transpose()], [0], 
                           shape=(self.c.shape[0], self.c.shape[0])).tocsr()

        return sparse.csc_matrix(D_c.power(-0.5).dot(self.V))

    def rpc(self):
        """Get row principal coordinates
        """
        
        D_s = sparse.diags([self.s], [0], 
                           shape=(self.s.shape[0], self.s.shape[0])).tocsr()

        return self.rsc().dot(D_s)

    def cpc(self):
        """Get column principal coordinates
        """
        
        D_s = sparse.diags([self.s], [0], 
                           shape=(self.s.shape[0], self.s.shape[0])).tocsr()

        return self.csc().dot(D_s)

    def pi(self):
        """Get the principal inertias
        """
        
        return self.s**2

    def scaled_pi(self):
        """
        Get the scaled principal inertias (broken)
        """
        
        return np.round(100 * self.pi() / self.inertia, 2)

    def suprow(self, row, principal=True):
        """ Calculate the position of a supplementary row

        inputs
        --

        row:         A 1xM array where M makes the vector conformable to the 
                     column principal coordinates
        principal:   principal coordinate (standard when false)
        
        outputs
        --
        A 1xK array of row principal coordinates

        """
        assert row.shape[0] == self.c.shape[0], 'Row not conformable'
        sr = sparse.csr_matrix(row)
        rprof = sr / sr.sum()
        if principal:   
            return rprof.dot(self.csc()).todense()
        else:
            f = rprof.dot(self.csc()).todense()
            D_sinv = sparse.diags([1/self.s], [0], 
                               shape=(self.s.shape[0], self.s.shape[0])).tocsr()

            return D_sinv.dot(f.T).T

    def supcol(self, col, principal=True):
        """ Calculate the position of a supplementary column

        inputs
        --

        col:       A 1xM array where M makes the vector conformable to the 
                   row principal coordinates
        principal: principal coordinate (standard coordinate when false)
        
        outputs
        --
        A 1xK array of column principal coordinates if principal=True, 
        standard coordinates otherwise

        """

        assert col.shape[1] == self.r.shape[0], 'Col not conformable'
        sc = sparse.csc_matrix(col)     
        cprof = sc / sc.sum()
        if principal:
            return cprof.dot(self.rsc())
        else:
            g = cprof.dot(self.rsc())
            D_sinv = sparse.diags([1/self.s], [0], 
                               shape=(self.s.shape[0], self.s.shape[0])).tocsr()

            return D_sinv.dot(g.T).T

    def __init__(self, D, cores=1, random_state=False, **kwargs):
        self._cores = cores
        self._seed = random_state
        self._began = datetime.today()
        self.U, self.s, self.V, self.r, self.c = self._cor_svd(D, **kwargs)
        self._completed = datetime.today()

    def summary(self):
        """Return text a summary of the correspondence analysis solution
        
        """
        
        label = 'CORRESPONDENCE ANALYSIS'
        padlen = round((80 - len(label))/2)
        began_str = datetime.strftime(self._began, '%Y-%m-%d %H:%m:%s')
        completed_str = datetime.strftime(self._completed, '%Y-%m-%d %H:%m:%s')
        duration = self._completed - self._began
        summarystr = '\n'.join([
            '-'*80,
            (' ' * padlen) + f'{label}' + (' ' * padlen),
            f'Began:    \t{began_str}\nCompleted:\t{completed_str}',
            f'(Completion time: {duration})',
            '-'*20,
            f'Total inertia: {round(self.inertia,4)}',
            'Principal inertias (eigenvalues):',
            'dim\tvalue\t    %\tcum %'])
        cumpct = 0
        for n, a in list(enumerate(zip(self.pi(), self.scaled_pi()))):
            cumpct += a[1]
            dimstr = f'{n+1:^3}\t{round(a[0],3):^5}\t{round(a[1],2):^5}\t{round(cumpct,1):^5}'
            summarystr += '\n'
            summarystr += dimstr
        summarystr += '\n'
        summarystr += '-'*80

        return summarystr

