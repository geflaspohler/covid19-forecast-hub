from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
import copy
from src.utils.general_util import tic, toc, printf, make_directories, symlink
import pdb 
from datetime import datetime, timedelta
from src.utils.models_util import get_forecast_filename
import os




class JointHinter():
    '''
    JointHinter modules generates a hint matrix H (dimension d x n) for d
    psuedohints provided by n hinters.
    '''
    def __init__(self, gt_id, hint_types, oe, regret_hints=False):
        ''' Initialize joint hinter
        Args:
           hint_types: list of list of strings specifying hint types for each horizon
                i.e., [["prev_y"], ["cfsv2", "doy", "uniform"], ["mean_g", "mean_y"]]
           oe: instance of OnlineExpert class with a loss_gradient function
           regret_hints: if False, returns sum of loss gradient hints. If True, returns
                sum of instantaneous expert regret hints
        '''
        self.D = len(hint_types) # number of horizon delay periods
        self.d = oe.d # dimension of the pseudogradients
        self.n = 0 # number of total hinters over horizons
        self.n_k = {} # horizon-specific number of hinters
        self.i_k = {} # start index of hinters
        self.hinters = {} # hinter object
        self.regret_hints = regret_hints

        # Initialize hinter objects
        for k in range(self.D):
            self.n_k[k] = len(hint_types[k])
            self.i_k[k] = self.n 
            self.n += self.n_k[k]
            self.hinters[k] = []
            for n_k in range(len(hint_types[k])):
                self.hinters[k].append(self.get_hinter(hint_types[k][n_k], oe, gt_id))

    def update_hint_data(self, g_fb, y_fb): 
        ''' Update each hinter with recieved feedback '''
        for k in range(self.D):
            for hinter in self.hinters[k]:
                hinter.update_hint_data(g_fb, y_fb)

    def reset_hint_data(self): 
        ''' Reset each hinters hint data '''
        for k in range(self.D):
            for hinter in self.hinters[k]:
                hinter.reset_hint_data()

    def get_hint_matrix(self, X_all, y_all, w_all, last_data_date, os_preds): 
        """ Get the d x n matrix of pseudo gradients

        Args:
           X_all: pd DataFrame containing expert predictions for target dates
           y_all: pd DataFrame containing ground truth values for dates in contest period
           w_all: pd DataFrame containing previous expert weights for dates in contest period
           last_data_date: DateTime object, signifying the last date (inclusive) for which
                ground truth data is available
           os_preds: list of DateTime objects, the outstanding prediction dates for
                the current expert.
            regret: If True provides hints in terms of instantaneous regrets for each expert.
                If False, returns hints in terms of accumlated loss gradients.
        """     
        if len(os_preds) > self.D:
            raise ValueError(f"{len(os_preds)} outstanding predictions for delay setting {self.D}")

        # Account for len(os_preds) < self.D at start
        hinter_index = list(range(self.D - len(os_preds), self.D))
        n_i = self.i_k[hinter_index[0]] # Get index of first hint in matrix

        # Initialize hint matrix
        hint_matrix = np.zeros((self.d, self.n))

        # Populate hint matrix
        for s in range(len(os_preds)):
            for hinter in self.hinters[hinter_index[s]]:
                # Get hint from correct hinter
                hint, hint_data = hinter.get_hint(
                    X_all, y_all, w_all, last_data_date, os_preds, D=s, regret=self.regret_hints)

                # Add hint as new column in hint matrix
                hint_matrix[:, n_i] = hint.reshape(self.d,)
                n_i += 1

        return hint_matrix 

    def get_hinter(self, hint_type, oe, gt_id):
        ''' Instantiate hinter according to hint_type '''
        # Get horizon paramter from hint type, if provided
        if type(hint_type) is tuple:
            horizon  = hint_type[1]
            hint_type = hint_type[0]   
        else:
            horizon = "34w" 

        if hint_type == "prev_y":
            hinter = PrevObs(oe)   
        elif hint_type == "mean_y":
            hinter = MeanObs(oe)    
        elif hint_type == "trend_y":
            hinter = TrendObs(oe)        
        # elif hint_type in ['catboost', 'cfsv2', 'doy', 'llr', 'multillr', 'salient_fri']: 
        #     ind = oe.expert_list.index(hint_type)
        #     if ind== -1:
        #         raise ValueError(f"Must provide {hint_type} in expert list {oe.expert_list}")
        #     expert_weights = np.zeros((len(oe.expert_list),))
        #     expert_weights[ind] = 1.0
        #     hinter = ExpertEnsemble(oe, expert_weights)
        elif hint_type in ['catboost', 'cfsv2', 'doy', 'llr', 'multillr', 'salient_fri']: 
            hinter = HorizonForecast(oe, hint_type, gt_id, horizon)
        elif hint_type == "uniform":
            expert_weights = np.ones((len(oe.expert_list),))
            expert_weights /= sum(expert_weights)
            hinter = ExpertEnsemble(oe, expert_weights)
        elif hint_type == "current_w":
            hinter = ExpertEnsemble(oe)                                   
        elif hint_type == "prev_g":
            hinter = PrevGrad(oe)                                   
        elif hint_type == "mean_g":
            hinter = MeanGrad(oe)    
        elif hint_type == "None":
            hinter = NoHint(oe)
        else:
            raise ValueError(f"Unrecognized hint type {hint_type}")
        return hinter

class Hinter(ABC):
    '''
    Hinter module mplements optimistic hinting utility for online learning.
    Abstract base class - must be instantiated with a particular hinting strategy
    '''
    def __init__(self, oe):
        """Initializes hinter 

        Args:
           oe: instance of OnlineExpert class with a loss_gradient function
        """     
        self.d = oe.d # number of experts
        self.loss_gradient= oe.loss_gradient # Function handle for the loss gradient
        self.reset_hint_data() # Initialize the hint data

    def most_recent_obs(self, y_all, last_data_date):
        """ Gets the most recent observation available up to last_data_date

        Args:
           y_all: a pd DataFrame containing ground truth values for dates in contest period
           last_data_date: DateTime object, signifying the last date (inclusive) for which
                ground truth data is available
        """  
        if y_all.index.get_level_values('target_date').isin([last_data_date]).any():                
            return y_all.loc[y_all.index.get_level_values('target_date') == last_data_date]
        else:
            # TODO: Efficiciency? 
            y_obs = y_all.loc[y_all.index.get_level_values('target_date') <= last_data_date]
            last_date = y_obs.tail(1).index.get_level_values('target_date')[0]
            printf(f"Warning: ground truth observation not avaliable on {last_data_date}")
            return y_all.loc[y_all.index.get_level_values('target_date') == last_date]
            # TODO: if there are missing gt values, could take the next most recent day

    def get_hint(self, X_all, y_all, w_all, last_data_date, os_preds, D=None, regret=False): 
        """ Gets the multi-day hint for the current expert

        Args:
           X_all: pd DataFrame containing expert predictions for target dates
           y_all: pd DataFrame containing ground truth values for dates in contest period
           w_all: pd DataFrame containing previous expert weights for dates in contest period
           last_data_date: DateTime object, signifying the last date (inclusive) for which
                ground truth data is available
           os_preds: list of DateTime objects, the outstanding prediction dates for
                the current expert.
            D: if None, return the set of pseudo-gradients and gradient sum over all
                outstanding predictions. Otherwise, D=0 returns the oldest outstanding
                prediction, D=1 the next, etc. 
            regret: If True provides hints in terms of instantaneous regrets for each expert.
                If False, returns hints in terms of accumlated loss gradients.
        """     
        grad_tilde = np.zeros((self.d,)) # multiday hints
        hint_data = {}

        # If insuffient data is avaliable for the hinter
        if (X_all is None) or (y_all is None):
            return grad_tilde, hint_data

        # Get outstanding date expert predictions
        X_os = X_all.loc[X_all.index.get_level_values('target_date').isin(os_preds)]
        i = 0
        for date, X in X_os.groupby(level='target_date'):
            if D is not None and D != i:
                i+=1
                continue

            # Always set the first value of the hint using the previous value of y, 
            # if this prev value is within one day of the oustanding prediction
            if i == 0 and (date - last_data_date).days <= 1:
                y_tilde = self.most_recent_obs(y_all, last_data_date).to_numpy()
                g_tilde = self.loss_gradient(X.to_numpy(), y_tilde, w=w_all.loc[date].to_numpy())
            else:
                g_tilde =  self.get_pseudo_grad(X, y_all, w_all, last_data_date, date)

            if regret: 
                # Return instantaneous regret
                w = w_all.loc[date].to_numpy()
                hint_data[date] = (np.dot(g_tilde, w) - g_tilde, w)
                grad_tilde += np.dot(g_tilde, w) - g_tilde
            else:
                # Return loss gradient
                hint_data[date] = (g_tilde, w_all.loc[date].to_numpy())
                grad_tilde += g_tilde       
            i+=1

        return grad_tilde, hint_data

    @abstractmethod
    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        """ Abstract method: gets the pseudo-gradient for the hint

        Args:
            X_all: pd DataFrame containing expert predictions for target dates
            y_all: pd DataFrame containing ground truth values for dates in contest period
            w_all: pd DataFrame containing previous expert weights for dates in contest period
            last_data_date: DateTime object, signifying the last date (inclusive) for which
                ground truth data is available
            date: DateTime object, date for which pseudo-gradient is applied
        """     
        pass

    @abstractmethod
    def update_hint_data(self, g_fb, y_fb): 
        """ Abstract method: updates any meta-data necessary to compute a hint

        Args:
            g_fb: a (d,) numpy array, containing a feedback gradient recieved by 
                the online learner
            w_fb: a (d,) numpy array, containing the weights played by the online
                learner, associated with g_fb
        """     
        pass

    @abstractmethod
    def reset_hint_data(self): 
        """ Abstract method: resets the hint data """     
        pass

class NoHint(Hinter):
    def __init__(self, oe):
        super().__init__(oe)

    def update_hint_data(self, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date):
        return np.zeros((self.d,)) # multiday hints

    def get_hint(self, X_all, y_all, w_all, last_data_date, os_preds, D=None, regret=False): 
        return np.zeros((self.d,)), {} # multiday hints

class HorizonForecast(Hinter):
    def __init__(self, oe, model, gt_id, horizon):
        super().__init__(oe)
        self.model = model
        self.horizon = horizon
        self.gt_id = gt_id

    def update_hint_data(self, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date):
        # Convert target date to string
        date_str = datetime.strftime(date, '%Y%m%d')  

        # Get names of submodel forecast files using the selected submodel
        fname = get_forecast_filename(model=self.model, 
                                                gt_id=self.gt_id,
                                                horizon=self.horizon,
                                                target_date_str=date_str)
        if not os.path.exists(fname):
            raise ValueError(f"No {self.model} forecast for horizon {self.horizon} on date {date_str}") 
        y_tilde = pd.read_hdf(fname).rename(columns={"pred": f"{self.model}"})
        y_tilde = y_tilde.set_index(['target_date', 'lat', 'lon']).squeeze().sort_index()
        
        # TODO: how expensive is to_numpy? Should we move to everything in pandas
        return  self.loss_gradient(X.to_numpy(), y_tilde.to_numpy(), w=w_all.loc[date].to_numpy()).reshape(-1,)

class ExpertEnsemble(Hinter):
    def __init__(self, oe, expert_weights=None):
        super().__init__(oe)
        self.ew = expert_weights

    def update_hint_data(self, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date):
        if self.ew is not None:
            y_tilde = X @ self.ew  # Use ew to get an estimate of w
        else: 
            w_curr = w_all.iloc[-1, :].to_numpy() # Get most recent w estimate
            y_tilde = X @ w_curr # Use ew to get an estimate of w

        # TODO: how expensive is to_numpy? Should we move to everything in pandas
        return  self.loss_gradient(X.to_numpy(), y_tilde.to_numpy(), w=w_all.loc[date].to_numpy()).reshape(-1,)

class PrevObs(Hinter):
    def __init__(self, oe):
        super().__init__(oe)

    def update_hint_data(self, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        # Get most recently available ground truth data
        y_tilde = self.most_recent_obs(y_all, last_data_date) 
        return self.loss_gradient(X.to_numpy(), y_tilde.to_numpy(), w=w_all.loc[date].to_numpy())

class MeanObs(Hinter):
    def __init__(self, oe):
        self.N = 514 # Dimension of observations
        super().__init__(oe)

    def update_hint_data(self, g_fb, y_fb): 
        if self.y_sum is None:
            self.y_sum = np.zeros(y_fb.shape)
        self.y_sum += y_fb
        self.y_len += 1

    def reset_hint_data(self): 
        self.y_sum = None
        self.y_len = 1

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        # Get most recently available ground truth data
        if self.y_sum is None:
            self.y_sum = np.zeros((self.N, ))
            # TODO: updgrade to get adaptively from y_all

        y_tilde = self.y_sum / self.y_len
        g_tilde = self.loss_gradient(X.to_numpy(), y_tilde, w=w_all.loc[date].to_numpy())
        return g_tilde

class TrendObs(Hinter):
    def __init__(self, oe):
        self.N = 514 # Dimension of observations
        super().__init__(oe)

    def update_hint_data(self, g_fb, y_fb): 
        self.y_prev[self.y_idx] = y_fb
        self.y_idx = (self.y_idx + 1) % 1

    def reset_hint_data(self): 
        self.y_prev = [np.zeros((self.N,)), np.zeros((self.N,))]    
        # TODO: get size automatically from somewhere
        self.y_idx = 0

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        # Get most recently available ground truth data
        next_idx = (self.y_idx + 1) % 1
        y_tilde = (self.y_prev[next_idx] - self.y_prev[self.y_idx]) + self.y_prev[next_idx]
        return self.loss_gradient(X.to_numpy(), y_tilde, w=w_all.loc[date].to_numpy())

class PrevGrad(Hinter):
    def __init__(self, oe):
        super().__init__(oe)

    def update_hint_data(self, g_fb, y_fb): 
        self.g_prev = g_fb

    def reset_hint_data(self): 
        self.g_prev = np.zeros((self.d,))

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        # Get most recently available ground truth data
        return self.g_prev

class MeanGrad(Hinter):
    def __init__(self, oe):
        super().__init__(oe)

    def reset_hint_data(self): 
        self.g_sum = np.zeros((self.d,))
        self.g_len = 0

    def update_hint_data(self, g_fb, y_fb): 
        self.g_sum += g_fb
        self.g_len += 1

    def get_pseudo_grad(self, X, y_all, w_all, last_data_date, date): 
        # Get most recently available ground truth data
        if self.g_len == 0:
            return self.g_sum 
        return self.g_sum / self.g_len

