import numpy as np

from Bifrost import Bifrost
from Kienzan import sigmoid


class Bomat(Bifrost):
    
    def __init__(self, n_variables=None, initialization='zeros', 
                 threshold=.5):
        super().__init__(n_variables, initialization)
        self.threshold = threshold
        
    def make_one_prediction(self, X_row):
        y_hat_i = self.coefficients['β0']
        for i, x in enumerate(X_row):
            y_hat_i += self.coefficients[f'β{i + 1}'] * x
        return sigmoid(y_hat_i)
    
    def make_decision(self, X):
        proba = self.make_prediction(X)
        decisions = []
        for y_hat_i in proba:
            if y_hat_i >= self.threshold:
                decisions.append(1)
            else:
                decisions.append(0)
        return np.array(decisions)