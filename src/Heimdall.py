import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from Heisenberg import mean_squared_error
from Heisenberg import binary_cross_entropy_loss


def stochastic_gradient_descent(X, y, model, coefficients, 
                                learning_rate, epochs, model_type: dict,
                                plot_results=False):
    '''
    Θj(t+1) := Θj(t) - α.[ (δ/δΘj).J(Θi,...,Θn) ] >
    Θj(t+1) := Θj(t) - α.[ J(Θi,...,Θn).xj ] 
    where J(Θi,...,Θn) is a cost function   
    
    coefficients: {'β0': v0, 'β1': v1, ..., 'βi': vi}    
    '''

    if plot_results:
        
        y_hat_init = model.make_prediction(X)
        
        def _plot_results(sample_rows, epoch_values):
                        
            plt.figure(figsize=(12, 6))
            plt.scatter(X[:, 0], y, label='real')
            plt.plot(X[:, 0], y_hat_init, label='inicialization')
            plt.plot(X, model.make_prediction(X[:, np.newaxis]), c='C3', 
                     label='grad-desc')
            plt.legend()
            plt.show();

            pd.DataFrame(epoch_values).apply(lambda x: (x-np.mean(x))/np.std(x)).plot(figsize=(15, 6))
            plt.plot([0 for r in range(len(epoch_values))])
            plt.show();

            plt.hist(sample_rows, bins=50)
            plt.show();        
            
            return 
    
    sample_rows, epoch_values = [], []
    for i in tqdm(range(epochs)):
        
        #get a random sample from the training data-set:
        row = random.randint(0, X.shape[0] - 1)
        sample_rows.append(row)
        
        # make a prediction:
        yhat = model.make_one_prediction(X[row, :])
        
        if list(model_type.keys())[0] == 'regression':
            # calculate the cost function for a single prediciton:
            cost = (-2)*sum((y[row, :] - yhat))
        elif list(model_type.keys())[0] == 'classification':
            # calculate the cost function for a single prediciton:
            cost = yhat - y[row, 0]
        
        # store coefficients and cost function values into epoch_values before update:
        iteration_values = coefficients.copy()
        if list(model_type.keys())[0] == 'regression':
            iteration_values['cost'] = mean_squared_error(y[row, :], yhat)
        elif list(model_type.keys())[0] == 'classification':
            iteration_values['cost'] = binary_cross_entropy_loss(y[row, :], yhat)
        epoch_values.append(iteration_values)
        
        # make coefficients update:
        coefficients['β0'] = coefficients['β0'] - (learning_rate * (cost * 1))
        for i, xi in enumerate(X[row, :]):
            coefficients[f'β{i + 1}'] = coefficients[f'β{i + 1}'] - (learning_rate * (cost * xi))
            
    # store lest update:
    iteration_values = coefficients.copy()
    if list(model_type.keys())[0] == 'regression':
        iteration_values['cost'] = mean_squared_error(y[row, :], yhat)
    elif list(model_type.keys())[0] == 'classification':
        iteration_values['cost'] = binary_cross_entropy_loss(y[row, :], yhat)
    epoch_values.append(iteration_values)
    
    if plot_results:
        _plot_results(sample_rows, epoch_values)
    
    return coefficients, (sample_rows, epoch_values)


def batch_gradient_descent(X, y, model, coefficients, 
                           learning_rate, epochs, plot_results=False):
    
    if plot_results:
        
        y_hat_init = model.make_prediction(X)
        
        def _plot_results(epoch_values):
                        
            plt.figure(figsize=(12, 6))
            plt.scatter(X[:, 0], y, label='real')
            plt.plot(X[:, 0], y_hat_init, label='inicialization')
            plt.plot(X, model.make_prediction(X[:, np.newaxis]), c='C3', 
                     label='grad-desc')
            plt.legend()
            plt.show();

            pd.DataFrame(epoch_values).apply(lambda x: (x-np.mean(x))/np.std(x)).plot(figsize=(15, 6))
            plt.plot([0 for r in range(len(epoch_values))])
            plt.show();
            
            return 
    
    epoch_values = []
    for i in tqdm(range(epochs)):
        
        # make prediction for all trainin set:
        yhat = model.make_prediction(X)
        
        # store coefficients and cost function values into epoch_values before update:
        iteration_values = coefficients.copy()
        iteration_values['cost'] = mean_squared_error(np.squeeze(y), yhat)
        epoch_values.append(iteration_values)
        
        # calculate the cost function for a single prediciton:
        delta_y = np.squeeze(y) - yhat
        cost = (-2/len(y))*sum(delta_y)
        
        # make coefficients update:
        coefficients['β0'] = coefficients['β0'] - (learning_rate * cost)
        for xi in range(X.shape[1]):
            cost = (-2/len(y))*sum(delta_y * X[:, xi])
            coefficients[f'β{xi + 1}'] = coefficients[f'β{xi + 1}'] - (learning_rate * cost)
            
    # store lest update:
    iteration_values = coefficients.copy()
    iteration_values['cost'] = mean_squared_error(np.squeeze(y), yhat)
    epoch_values.append(iteration_values)
    
    if plot_results:
        _plot_results(epoch_values)
            
    return coefficients, epoch_values


def mini_batch_gradient_descent(X, y, model, coefficients, 
                                learning_rate, epochs, batch_size, 
                                plot_results=False):
    
    if plot_results:
        
        y_hat_init = model.make_prediction(X)
        
        def _plot_results(epoch_values):
            
            plt.figure(figsize=(12, 6))
            plt.scatter(X[:, 0], y, label='real')
            plt.plot(X[:, 0], y_hat_init, label='inicialization')
            plt.plot(X, model.make_prediction(X[:, np.newaxis]), c='C3', 
                     label='grad-desc')
            plt.legend()
            plt.show();

            pd.DataFrame(epoch_values).apply(lambda x: (x-np.mean(x))/np.std(x)).plot(figsize=(15, 6))
            plt.plot([0 for r in range(len(epoch_values))])
            plt.show();
            
            return
    
    
    batch_size = round(X.shape[0] * batch_size)
    epoch_values = []
    for i in tqdm(range(epochs)):
        
        # generate the sample from X with size = batch_size        
        sample_index = np.random.choice([i for i in range(X.shape[0])], 
                                        size=batch_size, 
                                        replace=False)
        X_batch = X[sample_index, :]
        y_batch = y[sample_index, :]
        
        
        # make prediction for the batch set:
        yhat = model.make_prediction(X_batch)
        
        # store coefficients and cost function values into epoch_values before update:
        iteration_values = coefficients.copy()
        iteration_values['cost'] = mean_squared_error(np.squeeze(y_batch), yhat)
        epoch_values.append(iteration_values)
        
        # calculate the cost function for a single prediciton:
        delta_y = np.squeeze(y_batch) - yhat
        cost = (-2/len(y_batch))*sum(delta_y)
        
        # make coefficients update:
        coefficients['β0'] = coefficients['β0'] - (learning_rate * cost)
        for xi in range(X_batch.shape[1]):
            cost = (-2/len(y_batch))*sum(delta_y * X_batch[:, xi])
            coefficients[f'β{xi + 1}'] = coefficients[f'β{xi + 1}'] - (learning_rate * cost)
            
    # store lest update:
    iteration_values = coefficients.copy()
    iteration_values['cost'] = mean_squared_error(np.squeeze(y_batch), yhat)
    epoch_values.append(iteration_values)
   
    if plot_results:
        _plot_results(epoch_values)
     
    return coefficients, epoch_values