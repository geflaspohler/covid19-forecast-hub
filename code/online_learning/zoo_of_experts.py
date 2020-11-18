from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import mean_squared_error
import pandas as pd
import copy
from collections import deque
import pdb 
from src.utils.general_util import tic, toc, printf, make_directories, symlink
import zoo_of_losses
from functools import partial

def get_expert(alg, loss, T, expert_list, active_experts=None, init_method="uniform", reg=None, partition=None, init_df=None):
    if alg == "ftl":
        oe = FTL(loss, T, init_method, expert_list, partition=partition, active_experts=active_experts)    
    elif alg == "ftrl":
        oe = FTRL(loss, T, init_method, expert_list, init_df, reg, partition=partition, active_experts=active_experts)
    elif alg == "adahedgefo":
        oe = AdaHedgeFO(loss, T, init_method, expert_list, reg=reg, partition=partition, active_experts=active_experts)    
    elif alg == "rm":
        oe = RegretMatching(loss, T, init_method, expert_list, partition=partition, active_experts=active_experts)  
    elif alg == "rmplus":
        oe = RegretMatchingPlus(loss, T, init_method, expert_list, partition=partition, active_experts=active_experts)        
    elif alg == "adahedgesr":
        oe = AdaHedgeSR(loss, T, init_method, expert_list, partition=partition, active_experts=active_experts)    
    elif alg == "flip_flop":
        oe = FlipFlop(loss, T, init_method, expert_list, partition=partition, active_experts=active_experts)    
    else: # Unknown algorithm, instantiate abstract base class and throw error
        raise ValueError(f"Unknown expert type {alg}.")
    return oe

class OnlineExpert(ABC):
    """
    OnlineExpert module implements logic for online learning in the 
    forecast rodeo challenege, with various online learning algorithms.
    Abstract base class - must be instantiated with a particular algorithm. 
    """    
    def __init__(self, loss_obj, T, init_method, expert_list, partition=None):
        """Initializes online_expert 
        Args:
            T: integer > 0, duration of online learning
            init_method: weight initialization, one of "uniform", "doy"
            expert_list: list of strings indicating expert names
        """                
        # Check and store weight initialization 
        supported_meth = ['uniform', 'doy']
        if init_method not in supported_meth:
            raise ValueError(f"Unsupported method for weight initialization {params['init_method']}.")

        self.T = T # Algorithm horizon
        self.t = 1 # Current algorithm time

        self.method= init_method  # Weight initialization method
        self.expert_list = expert_list
        self.d = len(self.expert_list) # Number of experts

        # Initilize optimism hint, if hint is not "None"
        self.hint_prev = np.zeros(self.d,)

        # Initialize loss structure and set up function handels
        self.loss_obj = loss_obj
        self.loss = loss_obj.loss
        self.loss_experts = loss_obj.loss_experts
        self.loss_gradient = loss_obj.loss_gradient

        if partition is not None:
            self.partition = np.array(partition)
        else:
            self.partition = np.ones(len(self.expert_list),)
        self.partition_keys = set(self.partition)

        # Create regret_loss partial, prepopulated with partition information
        self.loss_regret = partial(loss_obj.loss_regret, self.partition, self.partition_keys)

    def log_params(self):
        """ Modifies online params in place with current algorithm parameters
            at target date string.
        """
        params = self.get_logging_params()
        return params

    def init_weight_vector(self, expert_list, method):
        """ Returns initialized weight vector. 

        Args:
           expert_list: a list of d string, corresponding to expert model names
           method: one of "uniform", "random", "doy". Initialization method for
               weight vector. 
        """          
        if method == 'uniform':
            w = 1./self.d * np.ones(self.d)
        elif method == 'doy':
            # Weights for non-doy models must be non-zero or exp gradients will fail       
            doy_i = expert_list.index("doy")
            doy_weight = 0.80 # Magic value
            if doy_i != -1:
                w = float((1-doy_weight)/self.d) * np.ones(self.d)
                w[doy_i] = doy_weight
            else:
                # Uniform initialization if doy is not found
                w = 1./d * np.ones(self.d)
        else:
            raise ValueError(f"Unsupported initialization method {method}.")        
        return w
    
    def get_weights(self):
        ''' Returns dictionary of expert model names and current weights '''
        return dict(zip(self.expert_list, self.w))
            
    def neg_entropy_reg(self, w):
        return np.sum(w[w>0.0] * np.log(w[w>0.0])) + np.log(self.d)
            
    def predict(self, X, hint=None): 
        pass
    
    def update_expert(self, X, y, w):
        pass

    @abstractmethod
    def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None): 
        pass

    @abstractmethod
    def get_logging_params(self):
        pass

    @abstractmethod
    def reset_alg_params(self, T, expert_df=None):
        pass    

class FTL(OnlineExpert):
    """
    Implements Follow the Leader online learning algorithm
    """    
    def __init__(self, loss_obj, T, init_method, expert_list, partition=None):
        """Initializes online_expert 
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)
        self.reset_alg_params(T)

    def reset_alg_params(self, T):
        """ Set hyperparameter values of online learner """ 
        #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
        self.t = 1
        self.T = T
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.ada_loss = np.zeros((1, self.d))

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            't': self.t
        }
        # Allow for duplicate model names in logging
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]

            params[alias] = float(self.w[i])
        return params

    def update_expert(self, X, y, w):
        """ Returns the next leader

        Args:
           X: G x self.d - prediction at G grid point locations from self.d experts
           y: G x 1 - ground truth forecast at G grid points
        """                        
        # Get gradient and update optimistc hints        
        g = self.loss_gradient(X, y, self.w)
        self.update_hint(X, y, update_pred=False)

        l_t = self.loss_experts(X, y)
        self.ada_loss += l_t

        leader = np.argmin(self.ada_loss)
        self.w = np.zeros((self.d, 1))
        self.w[leader] = 1.0

        self.t += 1      

    def predict(self, X, hint=None): 
        """Returns an ensemble of experts, given the current weights 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
        """        
        # Update 
        self.update_hint(X, update_pred=True)
        return X @ self.w

class FTRL(OnlineExpert):
    """
    FTRL module implements the Follow the Regularized Leader online learning algorihtm, with
    varying regularization step size formulations. 
    """    
    def __init__(self, loss, T, init_method, expert_list, init_df, reg="entropic", partition=None):
        """Initializes online_expert 
        Args:
           alg: online learning algorithm, one of: "forel", "adahedge", "adahedge_robust", "flip_flop"
           reg: online learning regularization, one of "quadratic", "entropic"
        """                

        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)

        # Store general parameters via base class
        self.L  = None # Initialized estimate of Lipschitz constant

        # Check and store regulraization 
        supported_reg = ['quadratic', 'entropic']
        if reg not in supported_reg:
            raise ValueError(f"Unsupported regularizer for FOTRL {reg}.")
        self.reg = reg

        self.reset_alg_params(T, init_df=init_df)

    def reset_alg_params(self, T, init_df=None):
        """ Set hyperparameter values of online learner """ 
        if init_df is None:
            raise ValueError("Must pass an expert_df for estimating global Lipschitz constants.")

        self.t = 1 # Reset online learning time
        self.T = T # Update planner duration
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.eta = self.init_step_size(init_df) # Algrorithm step size

    def init_step_size(self, T, expert_df):  
        """ Returns step size. If algorithm is forel, computes an estimate of the 
        Lipschitz constant using expert_df to compute fixed step size. If algorithm
        is adahedge, initializes step size to zero. 

        Args:
           expert_df: If provided, expert_df is used to computed empirical values 
               of L for forel algorithms. 
        """         
        if self.L is None:
            self.L = self.estimate_lipschitz_const(expert_df)

        # Set algorithm step size as reccomended by Shalev-Shwartz et al., #TODO Where
        if self.reg == 'quadratic':
            return 1./(self.L*np.sqrt(2*T))
        elif self.reg == 'entropic':           
            return np.sqrt(np.log2(self.d)) / (self.L*np.sqrt(2*T))
        else: # Default behavior 
            return 1./(self.L*np.sqrt(2*T))         

    def estimate_lipschitz_const(self, expert_df):
        """ Estimate the Lipschitz constant for different loss functions/regularizers
        Args:
           expert_df: a pd DataFrame used to computed empirical values 
               of L for setting the weight update step-size        
        """
        L = []
        # Remove mean value from each expert column; only valid for unit constraint on w
        for date, expert_pred in expert_df.groupby(level=0): # by target date
            X_tilde = expert_pred.to_numpy()
            X_tilde -= X_tilde.mean(axis=0)
            G = X_tilde.shape[0]
            if self.reg == 'quadratic':
                L.append(np.max(np.linalg.svd(X_tilde, compute_uv=False)) / G)
            elif self.reg == 'entropic':
                L.append(np.max(np.linalg.norm(X_tilde, axis=0, ord=2)) / G)                     
        return np.mean(L)         

    def get_logging_params(self):
        # Logging
        params = {
            'eta': self.eta,
            't': self.t
        }
        # Allow for duplicate model names in logging
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]

            params[alias] = float(self.w[i])
        
        return params

    def update_expert(self, X, y, w):
        """Performs one step of online gradient descent with respect to self.w. 
        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
           y: G x 1 np.array - ground truth forecast at G grid points
           w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
        """        
        # Get gradient and update optimistc hints        
        g = self.loss_gradient(X, y, self.w)
        self.update_hint(X, y, update_pred=False)

        if self.reg == "quadratic":
            self.w -= - self.eta*g
        elif self.reg == "entropic":
            # Remove min value from gradient for stable exponentiation
            z_stable = g - np.min(g, axis=None)
            g = self.w * np.exp(-self.eta * z_stable)
            self.w = g / np.sum(g, axis=None)
        self.t += 1      

    def predict(self, X, hint=None): 
        """Returns an ensemble of experts, given the current weights 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
        """        
        self.update_hint(X, update_pred=True)
        return X @ self.w

class AdaHedgeFO(OnlineExpert):
    """
    AdaHedge module implements AdaHedge online learning algorithm from Francesco Orabona's monograph
    """    
    def __init__(self, loss, T, init_method, expert_list, reg="orig", partition=None):
        """Initializes online_expert 
        Args:
           alg: online learning algorithm, one of: "forel", "adahedge", "adahedge_robust", "flip_flop"
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)

        # Check and store regulraization 
        supported_reg = ['orig', 'plusplus', 'delay_hint', 'delay_nohint', 'nodelay_hint', "forward_drift"]
        if reg not in supported_reg:
            raise ValueError(f"Unsupported regularizer for AdaHedgeFO {reg}.")
        self.reg = reg
        self.reset_alg_params(T)

    def reset_alg_params(self, T, expert_df=None):
        # Reset algorithm duration
        self.t = 1
        self.T = T

        # Initilaize algorithm hyperparmetersj
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.theta = np.zeros(self.w.shape) # The dual-space parameter 
        self.delta = 0.0 # per-iteration increase in step size
        self.Delta = 0.0 # cummulative step size increase
        self.eta = 0.0 # algorithm step-size or time varying regularization, called lambda_t in Orabona
        self.l_t = 0.0  # Cummulative loss of weights played
        self.alpha = np.log(self.d)
        self.tol = 1e-2

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            'delta': self.delta,
            'Delta': self.Delta,
            'lambda': self.eta,
            't': self.t
        }
        # Allow for duplicate model names in logging
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]

            params[alias] = float(self.w[i])
        return params

    def get_reg(self, w_eval, g_fb, hint, hint_prev, u_t, update_Delta=True):
        """ Returns the delayed optimistic foresight gains. If w_eval = self.w, returns
            optimistic feedback gains. If hint, hint_prev = 0, returns delayed feedback gains.
       
            w_eval: (d,) numpy array, location to evaluted loss 
            g_fb: (d,) numpy array, most recent feedback gradient
            hint: (d,) numpy array, pseudo-gradient hint at current timestep
            hint_prev : (d,) numpy array, pseudo-gradient hint at previous timestep
            u_t: (d,) numpy array, the best expert so far, one-hot encoding
                only required in the cost of optimistim
            set_Delta (boolean): if True, compute delta value and update self.Delta
                otherwise, use current value of self.Delta
        """
        if update_Delta:
            delta = self.get_delta(w_eval, g_fb, hint, hint_prev)
            Delta = self.Delta + delta  
        else:
            Delta = self.Delta
            delta = self.delta
            
        #$eta = (np.dot(hint, u_t - self.w) + Delta) / self.neg_entropy_reg(u_t)
        eta = (np.dot(hint, u_t - self.w) + Delta) / self.alpha

        # Enforce regularizer monotonicity
        return np.max([self.eta, eta]), Delta, delta

    def get_delta(self, w_eval, g_fb, hint, hint_prev):
        """ Returns the delayed optimistic foresight gains. If w_eval = self.w, returns
            optimistic feedback gains. If hint, hint_prev = 0, returns delayed feedback gains.
       
            w_eval: (d,) numpy array, location to evaluted loss 
            g_fb: (d,) numpy array, most recent feedback gradient
            hint: (d,) numpy array, pseudo-gradient hint at current timestep
            hint_prev : (d,) numpy array, pseudo-gradient hint at previous timestep
        """
        base = np.dot(w_eval, g_fb) + np.dot(self.w, hint - hint_prev) 
        maxv = np.max(g_fb + hint - self.hint_prev)
        if np.isclose(self.eta, 0):
            delta = base + np.max(self.theta - hint, axis=None) - \
                 np.max(self.theta + g_fb - hint_prev, axis=None)
        else:
            delta = base + self.eta * \
                np.log(np.sum(self.w * np.exp(-((g_fb + hint - self.hint_prev) - maxv) / self.eta))) 
        return delta

    def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None): 
        """Returns an ensemble of experts, given the current weights 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
        """  
        '''
        Incorporate loss from previous timestep
        '''  
        if X_fb is not None and y_fb is not None: 
            # Compute the loss of the previous play
            g_fb = self.loss_gradient(X_fb, y_fb, w_fb)  # linearize arround current w value        
            self.l_t += np.dot(g_fb, w_fb) # Loss of play
            #self.l_t = (self.l_t * (self.t -1)/self.t) + np.dot(g_fb, w_fb) / self.t # Loss of play, as a moving average
        else:
            # If no feedback is provided use zero gradient
            g_fb = np.zeros(self.theta.shape)
            w_fb = self.w

        if hint is None:
            hint = np.zeros(self.theta.shape)

        '''
        Update dual-space parameter value with standard gradient update
        '''
        self.theta = self.theta - g_fb 
        #self.theta = (self.theta * (self.t - 1)/self.t)  - g_fb / self.t # compute as a moving average 

        '''
        Update best expert so far
        '''

        # Set u to be the argmax of the hint
        maximizers = (self.theta == self.theta.max())
        h = copy.copy(hint)
        h[~maximizers] = -np.inf
        u_i = np.argmax(h)
        u_t = np.zeros((self.d,))
        u_t[u_i] = 1.0 

        '''
        u_t = np.zeros((self.d,))
        u_i = np.argmax(self.theta)
        u_t[u_i] = 1.0 
        '''
        
        '''
        if np.isclose(self.eta, 0):
            h = copy.copy(hint)
            h[~maximizers] = np.inf
            u_i = (h == h.min())
            u_t = np.zeros((self.d,))
            u_t[u_i] = 1.0 
        else:
            u_t = np.exp(-hint / self.eta) * maximizers 
        '''
        u_t = u_t / np.sum(u_t, axis=0)
        
        ''' 
        Get regularization value for current timestep
        '''
        zero = np.zeros(self.d,) # Zero hint dummy variable
        if self.reg == "orig":
            w_eval = self.w
            compute_Delta = True
            hc = zero # current hint

        elif self.reg == "delay_hint":
            w_eval = w_fb
            compute_Delta = True
            hc = hint # current hint

        elif self.reg == "nodelay_hint":
            w_eval = self.w
            compute_Delta = True
            hc = hint # current hint

        elif self.reg == "delay_nohint":
            w_eval = w_fb
            compute_Delta = True
            hc = zero # current hint
        elif self.reg == "plusplus":
            hc = hint
        elif self.reg == "forward_drift":
            hc = zero
        else:
            raise ValueError(f"Unrecognized regularization {self.reg}") 

        '''
        Update regularization
        '''
        if self.reg not in ["plusplus", "forward_drift"]:
            self.eta, self.Delta, self.delta  = self.get_reg(w_eval, g_fb, hc, self.hint_prev, u_t, compute_Delta)
        else:
            if np.isclose(self.eta, 0):
                Delta = np.inner(hc, u_t) + self.l_t + np.max(self.theta - hc, axis=None)
            else:
                maxv = np.max(self.theta - hc)
                Delta = np.inner(hc, u_t) + self.l_t + \
                    self.eta * np.log(np.sum(np.exp((self.theta - hc - maxv) / self.eta))) + maxv - self.eta * np.log(self.d)

            self.delta = Delta - self.Delta
            self.Delta = Delta
            self.eta = np.max([self.eta, self.Delta / self.alpha])
            
        '''
        Update expert weights 
        '''
        if np.isclose(self.eta, 0):
            w_i =  ((self.theta - hint) == (self.theta - hint).max()) # Set to the best expert so far
            self.w = np.zeros((self.d,))
            self.w[w_i] = 1.0  / np.sum(w_i)

        elif not np.isclose(self.eta, 0):
            maxv = np.max(self.theta - hint)
            self.w =  np.exp((self.theta - hint - maxv) / self.eta) 
            self.w = self.w / np.sum(self.w, axis=None)

        if np.isnan(self.w).any():
            pdb.set_trace()

        self.hint_prev = copy.deepcopy(hint)
        self.t += 1
        
        # Return prediction 
        return X_cur @ self.w

class AdaHedgeSR(OnlineExpert):
    """
    AdaHedge module implements AdaHedge online learning algorithm
    """    
    def __init__(self, loss,  T, init_method, expert_list, partition=None):
        """Initializes online_expert 
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)
        self.reset_alg_params(T)

    def reset_alg_params(self, T, expert_df=None):
        """ Set hyperparameter values of online learner """ 
        #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
        self.t = 1
        self.T = T
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.ada_loss = np.zeros((1, self.d))
        self.delta = 0.0
        self.eta = 0.0

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            'delta': self.delta,
            't': self.t
        }
        # Allow for duplicate model names in logging
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]
            params[alias] = float(self.w[i])
        return params

    def update_expert(self, X, y, w):
        """Performs one step of adahedge with respect to self.w. 

        Args:
           X: G x self.d - prediction at G grid point locations from self.d experts
           y: G x 1 - ground truth forecast at G grid points
           w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
        """                        
        # Get gradient and update optimistc hints        
        g = self.loss_gradient(X, y, self.w)
        self.update_hint(X, y, update_pred=False)

        if self.delta == 0:
            eta = np.inf
        else:
            eta = np.log(self.d) / self.delta
            
        w, Mprev = self.adahedge_mix(eta, self.ada_loss)
        
        l_t = self.loss_experts(X, y)
        h = w @ l_t.T
        
        self.ada_loss += l_t
        
        w, M = self.adahedge_mix(eta, self.ada_loss)
        self.w = w.T
        
        # Max clips numeric Jensen violation
        delta = np.max([0, h - (M - Mprev)])
        self.delta += delta
        
        self.t += 1  
        
    def adahedge_mix(self, eta, L):
        m = np.min(L)
        if (eta == np.inf):
            w = (L == m)
        else:
            w = np.exp(-eta * (L-m))
        
        s = np.sum(w, axis=None)
        w = w / s
        M = m - np.log(s / len(L)) / eta
        
        return w, M

    def predict(self, X, hint=None): 
        """Returns an ensemble of experts, given the current weights 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
        """        
        self.update_hint(X, update_pred=True)
        return X @ self.w

class RegretMatching(OnlineExpert):
    """
    Implements RegretMatching online learning algorithm
    """    
    def __init__(self, loss, T, init_method, expert_list=None, partition=None):
        """Initializes online_expert 
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)
        self.reset_alg_params(T)

    def reset_alg_params(self, T):
        """ Set hyperparameter values of online learner """ 
        #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
        self.t = 1
        self.T = T
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.regret = np.zeros((1, self.d))
        self.regret_hold = np.zeros((1, self.d))

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            't': self.t
        }
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]

            params[alias] = float(self.w[i])
        return params

    def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None): 
        """Performs one step of Regret Matching

        Args:
           X: G x self.d - prediction at G grid point locations from self.d experts
           y: G x 1 - ground truth forecast at G grid points
           w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
        """     

        '''
        Incorporate loss from previous timestep
        '''  
        if X_fb is not None and y_fb is not None: 
            # Compute the loss of the previous play
            g_fb = self.loss_gradient(X_fb, y_fb, w_fb)  # linearize arround current w value        

        else:
            # If no feedback is provided use zero gradient
            g_fb = np.zeros((self.d,))
            w_fb = self.w

        if hint is None:
            hint = np.zeros((self.d,))

        '''
        Update regret
        '''
        self.regret += np.dot(g_fb, w_fb) - g_fb
        printf(f"regret: {self.regret}")
        regret_pos = np.maximum(0, self.regret + hint)

        '''
        Update expert weights 
        '''
        if np.sum(regret_pos) > 0.0:
            self.w = (regret_pos / np.sum(regret_pos)).reshape(self.d,)
        else:
            self.w = 1./self.d * np.ones(self.d) # Uniform
        
        self.hint_prev = copy.deepcopy(hint)
        self.t += 1

        # Return prediction 
        return X_cur @ self.w

class RegretMatchingPlus(OnlineExpert):
    """
    Implements RegretMatching+ online learning algorithm
    """    
    def __init__(self, loss, T, init_method, expert_list=None, active_experts=None, partition=None):
        """Initializes online_expert 
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)
        self.reset_alg_params(T, active_experts)

    def reset_alg_params(self, T, active_experts):
        """ Set hyperparameter values of online learner """ 
        self.t = 1 # Current prediction iteration
        self.T = T # Duration of prediction period

        # State vector for RM+, mirror descent. Lives in the orthant
        self.w = np.zeros((self.d,)) # Must initialize weight vector to zero

        # Set of active experts for initial prediction.  Default to all active experts.
        self.prev_active_experts = np.zeros((self.d,), dtype=bool)
        if active_experts is None:
           self.active_experts = np.ones((self.d,), dtype=bool)
        else:
            self.active_experts = np.array(active_experts, dtype=bool)

        # Convex combination; derived from w, initialized to uniform over active experts
        self.p = np.zeros((self.d,)) #
        self.p[self.active_experts] = 1./self.d

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            't': self.t
        }
        for i in range(self.d):
            params[self.expert_list[i]] = float(self.p[i])
        return params

    def update_and_predict(self, X_cur, active_ind=None, hint=None, X_fb=None, y_fb=None, w_fb=None): 
        """Performs one step of Regret Matching+

        Args:
           X: 1 x self.d - prediction from self.d experts
           y: 1 x 1 - ground truth value 
           w: self.d x 1 np.array - prediction weights (convex combination) used 
                in round when X was expert and y was gt
        """     
        '''
        Get the set of currently active experts
        '''
        if active_ind is None:
            active_ind = np.arange(self.d)

        self.prev_active_experts = self.active_experts.copy()
        self.active_experts[active_ind] = True

        '''
        Incorporate loss from previous timestep
        '''  
        if X_fb is not None and y_fb is not None: 
            # Compute the loss of the previous play
            g_fb = self.loss_gradient(X_fb, y_fb, w_fb)  # linearize arround current w value        
            regret_fb = self.loss_regret(g_fb, w_fb, active_experts=self.active_experts) # compute regret w.r.t. partition
        else:
            # If no feedback is provided use zero gradient
            g_fb = np.zeros((self.d,))
            self.p = np.zeros((self.d,)) #
            self.p[self.active_experts] = 1./self.d
            w_fb = self.p
            regret_fb = np.zeros((self.d,))

        if hint is None:
            hint = np.zeros((self.d,))
        '''
        Update regret
        '''
        self.w = np.maximum(0, self.w + regret_fb + hint - self.hint_prev).reshape(-1,)

        '''
        Update expert weights 
        '''
        #  Get updated weight vector
        self.p = np.zeros((self.d,))
        for k in self.partition_keys:     
            p_ind = (self.partition == k) & (self.active_experts)
            if np.sum(self.w[p_ind]) > 0.0:
                self.p[p_ind] = (self.w[p_ind]/ np.sum(self.w[p_ind]))
            else:
                n_k = sum(p_ind)
                self.p[p_ind] = 1./n_k * np.ones(n_k,) # Uniform

        self.hint_prev = copy.deepcopy(hint)
        self.t += 1

        # Return prediction 
        return X_cur @ self.p

class FlipFlop(AdaHedgeSR):
    def __init__(self, loss, T, init_method, expert_list, partition=None):
        """Initializes online_expert 
        """                
        # Base class constructor 
        super().__init__(loss, T, init_method, expert_list, partition)
        self.reset_alg_params(T)

    def reset_alg_params(self, T, expert_df=None):
        """ Set hyperparameter values of online learner """ 
        #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
        self.t = 1
        self.T = T
        self.w = self.init_weight_vector(self.expert_list, self.method) # Expert weight vector
        self.ada_loss = np.zeros((1, self.d))
        self.delta = np.zeros((2,))
        self.alpha = 1.243
        self.phi = 2.37
        self.regime = 0
        self.scale = np.array([self.phi/self.alpha, self.alpha])
        self.eta = 0.0

    def get_logging_params(self):
        # Update parameter logging 
        params = {
            'regime': self.regime,
            't': self.t
        }
        dup_count = 1
        for i in range(self.d):
            if self.expert_list[i] in params:
                alias = self.expert_list[i] + str(dup_count)
                dup_count += 1
            else:
                alias = self.expert_list[i]

            params[alias] = float(self.w[i])
        return params
    
    def update_expert(self, X, y, w):
        """Performs one step of RegretMatching+

        Args:
           X: G x self.d - prediction at G grid point locations from self.d experts
           y: G x 1 - ground truth forecast at G grid points
           w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
        """                        
        # Get gradient and update optimistc hints        
        g = self.loss_gradient(X, y, self.w) # Take gradient at current self.w
        self.update_hint(X, y, update_pred=False)

        if self.regime == 0 or self.delta[1] == 0.0 :
            eta = np.inf
        else:
            eta = np.log(self.d) / self.delta[1]

        w, Mprev = self.adahedge_mix(eta, self.ada_loss)

        l_t = self.loss_experts(X, y)
        h = w @ l_t.T
        
        self.ada_loss += l_t
        w, M = self.adahedge_mix(eta, self.ada_loss)
        self.w = w.T        
        
        # Max clips numeric Jensen violation
        delta = np.max([0, h - (M - Mprev)])
        self.delta[self.regime] += delta                    

        if self.delta[self.regime] > self.scale[self.regime] + self.delta[1-self.regime]:
            self.regime = 1 - self.regime   

        self.t += 1

    def predict(self, X, hint=None): 
        """Returns an ensemble of experts, given the current weights 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts
        """        
        self.update_hint(X, update_pred=True)
        return X @ self.w
