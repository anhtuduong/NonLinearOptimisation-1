import numpy as np 
import pandas as pd

# ----------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------

def load_data(data_file):
    """_summary_

    Args:
        data_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = pd.read_excel(data_file)
    y = np.array(data['survived'])
    X = data.drop(columns=['name', 'survived'])
    X = X.values
    return y, X

def func_loglikelihood(wbar, X, y):
    """_summary_

    Args:
        wbar (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        wbar (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        wbar (_type_): _description_
        X (_type_): _description_

    Returns:
        _type_: _description_
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