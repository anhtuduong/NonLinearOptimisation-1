import numpy as np 
import pandas as pd

data = pd.read_excel('D:/Block 5/titanic.xlsx')
y = np.array(data['survived'])

X = data.drop(columns=['name', 'survived'])
X = X.values

def func_loglikelihood(wbar, X, y) :
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    firstPart = y * equation
    secondPart = np.log(1 + np.exp(equation))
    loglikelihoodFunction = np.sum(firstPart - secondPart)

    return loglikelihoodFunction

likelihood = func_loglikelihood([1, 1], X, y)
print (likelihood)

def grad_loglikelihood(wbar, X, y) :
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    phi = 1 / (1 + np.exp(-equation))
    array = np.array( 1 ,X )
    gradient = np.sum((y - phi) @array )
    
    return gradient

def hes_loglikelihood(wbar, X) :
    w0 = wbar[0]
    w = np.delete(wbar, 0)
    wT= np.transpose(w)
    equation = w0 + wT @ X
    
    array = np.array( 1 ,X )
    arrayTranspose = np.transpose(array)
    phi = 1 / (1 + np.exp(-equation))
    
    hessian = -np.sum((phi*(1 - phi))*array@arrayTranspose)
    
    return hessian



