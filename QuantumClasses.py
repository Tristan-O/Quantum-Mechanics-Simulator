## --- import statements --- ##
import numpy as np
import matplotlib.pyplot as plt
import types

class Operator:
    '''Basically a list of functions to do, in order from left to right'''
    hbar = 1
    def Identity(  ):
        return Operator( lambda a_state : State(lambda x : a_state.x_rep_func(x),a_state[0]) )
    def Position(  ):
        return Operator( lambda a_state : State(lambda x : x*a_state.x_rep_func(x),a_state[0]) )
    def Momentum(  ):
        return Operator( lambda a_state : State(lambda x : -1j* (Operator.hbar) * np.gradient( a_state.x_rep_func(x), x),a_state[0]) )
    def Momentum2(  ):
        return Operator.Momentum()**2 
    
    def __init__(self, f,verbose=False):
        '''Give an operation to eventually perform on a state'''
        # https://stackoverflow.com/a/6528148
        self.operate = types.FunctionType(f.__code__, f.__globals__, f.__name__,
                                          f.__defaults__, f.__closure__)
        self.mat = None
        self.x_used_for_mat = None
        self.gen_basis = lambda n,a : (lambda x,m=n: np.exp(2j* m *np.pi*x/a)/(a)**0.5 )
        self.verbose = verbose
    
    def debug(self, *args):
        if self.verbose:
            print("DEBUG:", args)
            
    def to_mat(self,x=None,N=500,diag=False, ret=False):
        '''Produces the matrix equivalent of this operator, in the basis of cos(nx)
        If diag is True, it will attempt to also diagonalize this matrix.'''
        self.debug('Making matrix: ', (2*N+1,2*N+1))
        if x is None and self.x_used_for_mat is None:
                raise ValueError('You must provide an x array to use to calculate the matrix over')
        a = x[-1]-x[0]
        self.mat = np.zeros((2*N+1,2*N+1),dtype=complex)
        self.x_used_for_mat = np.copy(x)
        
        op_basisState = np.zeros(2*N+1,dtype=State)
        basisState    = np.zeros(2*N+1,dtype=State)
        for i in range(-N,N+1):
            op_basisState[i+N] = self*State( self.gen_basis(i,a),x )
            basisState[i+N]    =      State( self.gen_basis(i,a),x )  
        self.debug('Basis States Made')
        self.mat = np.array(np.outer(basisState,op_basisState),dtype=complex) #vectorize this to make it faster
        self.debug('Matrix is ', np.shape(self.mat))
        if ret:
            return np.copy(self.mat)
    
    def find_eig(self,x=None,N=500,retStates=False):
        '''Find eigenvals and eigenvectors. The larger N is, the more accurate results will be.'''
        if self.mat is None:
            self.to_mat(x=x,N=N)
        elif np.shape(self.mat) != (2*N+1,2*N+1):
            self.to_mat(x=x,N=N)
        eig_vals,eig_vecs = np.linalg.eig(self.mat)
        if retStates:
            eig_states = []
            for vec in eig_vecs:
                eig_states.append( sum([ State() ]))
                
        return eig_vals,eig_vecs

    def __mul__(self, nxtobj):
        '''Define how multiplication works on another object'''
        if isinstance(nxtobj, Operator):
            return Operator( lambda a_state : self.operate(nxtobj.operate(a_state)) )
        elif isinstance(nxtobj, int) or isinstance(nxtobj, float) or isinstance(nxtobj, complex):
            return Operator( lambda a_state : nxtobj*self.operate(a_state) )
        elif isinstance(nxtobj, State):
            return self.operate( nxtobj )
        else:
            raise ValueError
    def __rmul__(self, prevobj):
        if isinstance(prevobj, int) or isinstance(prevobj, float) or isinstance(prevobj, complex):
            return Operator( lambda a_state : prevobj*self.operate(a_state) )
        elif isinstance(prevobj, State):
            raise ValueError('Be more careful with parentheses!')
    def __add__(self,nxtobj):
        '''Define how multiplication works on another object'''
        if isinstance(nxtobj, Operator):
            return Operator( lambda a_state : self.operate(a_state) + nxtobj.operate(a_state) )
        elif isinstance(nxtobj, int) or isinstance(nxtobj, float) or isinstance(nxtobj, complex):
            return Operator( lambda a_state : nxtobj*Operator.Identity(a_state) + self.operate(a_state) )
        else:
            raise ValueError
    def __sub__(self,nxtobj):
        return self + (-1)*nxtobj
    def __pow__(self,n):
        n = int(n)
        op = Operator.Identity()
        for i in range(n):
            op *= self
        return op

class State:
    def __init__(self, f, x):
        self.x_rep_func = types.FunctionType(f.__code__, f.__globals__, f.__name__,
                                             f.__defaults__, f.__closure__)
        self.x_arr = np.copy(x)
    
    def conjugate(self):
        return State( lambda x : np.conj(self.x_rep_func(x)), self.x_arr )
        
    def norm(self):
        '''Normalize this state'''
        amplitude = (self*self)**0.5
        f = self.x_rep_func
        self.x_rep_func = lambda x : (1/amplitude)*types.FunctionType(f.__code__, f.__globals__, f.__name__,
                                      f.__defaults__, f.__closure__)(x)
        return self
    
    def plot(self, ax=None,labelR='Real Part',labelI='Imaginary Part',subplots=False):
        '''Plot the real and imaginary parts of this state'''
        if ax is None:
            ax = plt.gca()
        y = self.x_rep_func(self.x_arr)
        pR = ax.plot(self.x_arr,  np.real(y), label=labelR)
        pI = ax.plot(self.x_arr,  np.imag(y), label=labelI)
        ax.legend()
        return pR,pI
    
    def __mul__(self, nxtobj):
        '''Define A|a_state> and <b_state|a_state>'''
        if isinstance(nxtobj, int) or isinstance(nxtobj, float) or isinstance(nxtobj, complex):
            return State( lambda x : nxtobj*self.x_rep_func(x), self.x_arr )
        elif isinstance(nxtobj, State):
            if np.shape(self[0])!=np.shape(nxtobj[0]) or self[0][0]!=nxtobj[0][0] or self[0][-1]!=nxtobj[0][-1]:
                raise ValueError
            return np.trapz(self.conjugate()[1] * nxtobj[1],self.x_arr) #integrate (psi*)(psi)
        else:
            raise ValueError
    def __rmul__(self, prevobj):
        if isinstance(prevobj, int) or isinstance(prevobj, float) or isinstance(prevobj, complex):
            return State( lambda x : prevobj*self.x_rep_func(x), self.x_arr )
    def __add__(self, nxtobj):
        if isinstance(nxtobj, State):
            return State( lambda x : self.x_rep_func(x)+nxtobj.x_rep_func(x), self.x_arr)
        else:
            raise ValueError
    def __getitem__(self,i):
        '''Returns copies of numpy arrays stored in this state
        [0] returns x_arr
        [1] returns wavefunction psi(x)'''
        if i == 0:
            return np.copy( self.x_arr )
        elif i == 1:
            return self.x_rep_func( self.x_arr )
        else: 
            raise ValueError