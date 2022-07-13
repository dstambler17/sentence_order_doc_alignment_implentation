import scipy.special as sc
import numpy as np

from mcerp import PERT

PEAKEDNESS_PARAM = 20
NUM_TIME_SLOTS = 16


class WindowingDistribution():
    '''
    Define a windowing distribution, Parent class
    '''
    def __init__(self, min_val: int, max_val: int):
        self.min = min_val
        self.max = max_val
        self.mode = None

    def evaluate_distribution(self, x):
        raise NotImplemented
    
    def get_mode():
        return mode

    def get_mean():
        raise NotImplemented
    
    def get_varience():
        raise NotImplemented


class ModifiedPert(WindowingDistribution):
    '''
    mode = ((j + 0.5) / J ) * N
    j = window number
    J = number of "sub-doc vectors"
    N = num sentences
    gamma = hyperparam for calculating mean

    Need to calculate PDF

    Then given an n, can eval a y
    More detail, see:
    https://www.vosesoftware.com/riskwiki/ModifiedPERTdistribution.php#
    :~:text=This%20modified%20PERT%20distribution%20can,his%2Fher%20opinion%20most%20accurately.
    '''        
    def __init__(self, min_val, large_J_val, num_sentences, gamma=PEAKEDNESS_PARAM):       
        #Define pert specific params
        self.gamma = gamma
        self.num_sentences = num_sentences
        self.large_J_val = large_J_val
        self.j_val = None
        super().__init__(min_val, num_sentences)

    
    def _calculate_mode(self, j, J, N):
        """
        Find mode as specified in paper
        """
        return ((j + 0.5) / J) * N

    
    def evaluate_distribution(self, x, j_val):
        '''
        Pert PDF function
        '''

        gamma = self.gamma #kwargs.get('gamma', 0)
        if j_val != self.j_val:
            self.j_val = j_val
            self.mode = self._calculate_mode(j_val, self.large_J_val, self.num_sentences)
        
        def _alpha_one():
            '''
            Helper function for pdf. Calculates alpha_one
            '''
            return 1 + (gamma * ((self.mode - self.min) / (self.max -self.min) ) )
    
        def _alpha_two():
            '''
            Helper function for pdf. Calculates alpha_two
            '''
            return 1 + (gamma * ((self.max - self.mode) / (self.max -self.min)) )
        
        def _beta_func(alpha_one, alpha_two):
            '''
            Helper function for pdf. Calculates beta_func
            '''
            return sc.beta(alpha_one, alpha_two)
            
        alpha_one, alpha_two = _alpha_one(), _alpha_two()
        beta_func_output = _beta_func(alpha_one, alpha_two)

        #first factor of numerator, second factor of numerator
        #second factor of denominator. First factor of denom is beta func output
        numerator_one = (x - self.min)**(alpha_one - 1)
        numerator_two = (self.max - x)**(alpha_two - 1)
        denom_two = (self.max - self.min)**(alpha_one + alpha_two - 1)

        return (numerator_one * numerator_two) / (beta_func_output * denom_two)

    
    def get_mean():
        return (self.min + (self.gamma * self.mode) + self.max)/(self.gamma + 2)
    
    def get_varience():
        mean = self.get_mean()
        return ((mean - self.min) * (self.max - mean))/(self.gamma + 3)


class ModifiedPertV2(WindowingDistribution):
    '''
    Uses a pert library to build doc vector
    '''
    def __init__(self, min_val, num_sentences, num_slots=NUM_TIME_SLOTS, gamma=PEAKEDNESS_PARAM):       
        #Define pert specific params
        self.gamma = gamma
        self.num_sentences = num_sentences

        self._build_pert_cache(num_slots)
        super().__init__(min_val, num_sentences)

    def _build_pert_cache(self, num_slots):
        '''
        Build cache to avoid major slow downs
        '''
        cache_size = 100
        _xx = np.linspace(start=0, stop=1, num=num_slots)
        self.PERT_CACHE = []
        for _pp in np.linspace(0, 1, num=cache_size):
            if _pp == 0.5:  # some special case that makes g do nothing
                _pp += 0.001
            pert = PERT(low=-0.001, peak=_pp, high=1.001, g=self.gamma, tag=None)
            _yy = pert.rv.pdf(_xx)
            _yy = _yy / sum(_yy)  # normalize
            self.PERT_CACHE.append(_yy)
    
    def get_cache_idx(self, cache_idx):
        return self.PERT_CACHE[cache_idx]


        



if __name__ == "__main__":
    #Test modified Pert
    mp = ModifiedPert(0, 16, 60)
    print(mp.evaluate_distribution(10, 2))
    print(mp.evaluate_distribution(9, 2))
    print(mp.evaluate_distribution(20, 2))
    print(mp.evaluate_distribution(0, 2))
    print(mp.evaluate_distribution(30, 2))
    print(mp.evaluate_distribution(40, 2))

    print("Test!!!")
    print(mp.evaluate_distribution(0.1, 0))
    print(mp.evaluate_distribution(1, 0))
    print(mp.evaluate_distribution(2, 0))
    print(mp.evaluate_distribution(9, 0))
    print(mp.evaluate_distribution(10, 0))
    print(mp.evaluate_distribution(20, 0))
    print(mp.evaluate_distribution(30, 0))
    print(mp.evaluate_distribution(40, 0))