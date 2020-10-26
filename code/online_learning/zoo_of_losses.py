from abc import ABC, abstractmethod
import numpy as np
import copy
import pdb 
from sklearn.metrics import mean_squared_error

from src.utils.general_util import tic, toc, printf, make_directories, symlink

class OnlineLoss(ABC):
    """ Abstract Loss object for online learning. """ 
    def __init__(self):
        pass

    def loss_regret(self, partition, partition_keys, g, w):
        ''' Computes the loss w.r.t. a partition 
        Args:
            partition: np Array masks for partitions of the vectors [1, 1, 2, 3, 3]
            partition_keys: list of unique partition keys
            g: np Array containing gradient
            w: np Array containing weight vector
        '''
        regret = np.zeros(g.shape)
        for k in partition_keys:
            p_ind = (partition == k)
            regret[p_ind] = np.dot(g[p_ind], w[p_ind]) - g[p_ind] 
        return regret

    @abstractmethod
    def loss(self, y, y_hat):
        pass

    @abstractmethod
    def loss_experts(self, X, y):
        pass

    @abstractmethod
    def loss_gradient(self, X, y, w):
        pass

class MAELoss(OnlineLoss):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        """Computes the mean absolute error 

        Args:
           y: 1 x 1 np.array - ground truth at G grid points
           y_hat: 1 x 1 np.array - forecast at G grid points

        """     
        return np.abs(y - y_hat)
    
    def loss_experts(self, X, y):
        """Computes the mean absolute error. 

        Args:
           X: 1 x self.d np array - prediction from self.d experts        
           y: 1 x 1 np.array - ground truth 
        """     
        d = X.shape[1]
        return np.abs(X - y)
    
    def loss_gradient(self, X, y, w):
        """Computes the gradient of the rodeo RMSE loss at location w. 

        Args:
           X: G x d np array - prediction at G grid point locations from self.d experts
           y: G x 1 np.array - ground truth at G grid points
           w: d x 1 np.array - location at which to compute gradient.
        """
        d = X.shape[1] # Number of experts 

        err = X @ w - y
        if np.isclose(err, np.zeros(err.shape)).all():
            return np.zeros((d,))
        elif err > 0:
            return err
        elif err < 0:
            return -err

class RodeoLoss(OnlineLoss):
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           y: G x 1 np.array - ground truth at G grid points
           y_hat: G x 1 np.array - forecast at G grid points

        """     
        return np.sqrt(mean_squared_error(y, y_hat))
        #return np.sqrt(np.mean(y - y_hat, axis=0))    
    
    def loss_experts(self, X, y):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X: G x self.d np array - prediction at G grid point locations from self.d experts        
           y: G x 1 np.array - ground truth at G grid points
        """     
        d = X.shape[1]
        return np.sqrt(np.mean((X - np.matlib.repmat(y.reshape(-1, 1), 1, d))**2, axis=0))    
    
    def loss_gradient(self, X, y, w):
        """Computes the gradient of the rodeo RMSE loss at location w. 

        Args:
           X: G x d np array - prediction at G grid point locations from self.d experts
           y: G x 1 np.array - ground truth at G grid points
           w: d x 1 np.array - location at which to compute gradient.
        """
        G = X.shape[0] # Number of grid points
        d = X.shape[1] # Number of experts 

        err = X @ w - y

        if np.isclose(err, np.zeros(err.shape)).all():
            return np.zeros((d,))

        #TODO: might be faster to compute (err.T @ X).T; also may not matter!
        return (X.T @ err / (np.sqrt(G)*np.linalg.norm(err, ord=2))).reshape(-1,)


class HintingLossTwoNorm(OnlineLoss): 
    def __init__(self):
        self.q = 2 # TODO: extend to support other norms

    def loss(self, y, y_hat):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           y: d x 1 np.array - true loss gradient
           y_hat: d x 1 np.array - predicted loss gradient 
        """     
        return 0.5 * np.linalg.norm(y - y_hat, ord=q)**2
    
    def loss_experts(self, X, y):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X: d x n np array - prediction at d gradient values from n experts        
           y: d x 1 np.array - ground truth gradient value at d points
        """    
        n = X.shape[1] 
        return np.sum((X - np.matlib.repmat(y.reshape(-1, 1), 1, n))**2, axis=0)
    
    def loss_gradient(self, X, y, w):
        """Computes the gradient of the rodeo RMSE loss at location w. 

        Args:
           X: d x n np array - prediction at G grid point locations from self.d experts
           y: d x 1 np.array - ground truth at G grid points
           w: n x 1 np.array - location at which to compute gradient.
        """
        d = X.shape[0] # Number of gradient values 
        n = X.shape[1] # Number of experts
        
        err = X @ w - y

        if np.isclose(err, np.zeros(err.shape)).all():
            return np.zeros((n,))

        return (X.T @ err).reshape(-1,)

