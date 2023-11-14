import numpy as np


class Bifrost():
    
    def __init__(self, n_variables=None, initialization='random'):
        '''
        create βi's regression coefficients for X variables plus β0.
        '''
        if n_variables:
            if initialization == 'random':
                self.coefficients = {'β0': np.random.random()}
                for n in range(n_variables):
                    self.coefficients[f'β{n + 1}'] = np.random.random()
            if initialization == 'zeros':
                self.coefficients = {'β0': 0}
                for n in range(n_variables):
                    self.coefficients[f'β{n + 1}'] = 0                
        else:
            self.coefficients = {}

    def make_one_prediction(self, X_row):
        y_hat_i = self.coefficients['β0']
        for i, x in enumerate(X_row):
            y_hat_i += self.coefficients[f'β{i + 1}'] * x
        return y_hat_i

    def make_prediction(self, X):
        '''
        return y_hat = f(X) = β0 + β1.x1 + ... + βn.xn
        '''
        y_hat = []
        for X_row in X:
            y_hat_i = self.make_one_prediction(X_row)
            y_hat.append(y_hat_i)
        return np.array(y_hat)