import numpy as np 
import pandas as pd

# ----------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------

def load_data(data_file):
    """ Load data from excel file
    and return vector y and matrix X

    Args:
        data_file (str): path to data file

    Returns:
        y (np.array): vector of label 'survived'
        X (np.array): matrix of features
    """
    data = pd.read_excel(data_file)
    y = np.array(data['survived'])
    X = data.drop(columns=['name', 'survived'])
    X = X.values
    return y, X

def func_loglikelihood(wbar, X, y):
    """ Compute the log-likelihood function

    Args:
        wbar (np.array): vector of weights
        X (np.array): matrix of features
        y (np.array): vector of label 'survived'

    Returns:
        loglikelihoodFunction (): log-likelihood function
    """
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    firstPart = y * equation
    secondPart = np.log(1 + np.exp(equation))
    loglikelihoodFunction = np.sum(firstPart - secondPart)

    return loglikelihoodFunction

def grad_loglikelihood(wbar, X, y):
    """

    """
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    phi = 1 / (1 + np.exp(-equation))
    array = np.array( 1 ,X )
    gradient = np.sum((y - phi) @array )
    
    return gradient

def hes_loglikelihood(wbar, X):
    """
    
    """
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    array = np.array( 1 ,X )
    arrayTranspose = np.transpose(array)
    phi = 1 / (1 + np.exp(-equation))
    
    hessian = -np.sum((phi*(1 - phi))*array@arrayTranspose)
    
    return hessian

# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------

if __name__ == '__main__':

    data_file = 'titanic.xlsx'
    y, X = load_data(data_file)