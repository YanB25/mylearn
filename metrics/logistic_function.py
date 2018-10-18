import numpy as np
def ReLu(dataset):
    '''
    ReLu function. y=x when x > 0, else y = 0.
    @param dataset :: Iterable at any shape
    '''
    return np.where(dataset > 0, dataset, 0)

def LeakReLu(slope, dataset):
    '''
    ReLu function. y=x when x > 0, else y = -kx.
    @param slope :: Float, the slope k.
    @param dataset :: Iterable at any shape
    '''
    return np.where(dataset > 0, dataset, dataset * slope)
def Sigmoid(dataset):
  return 1 / (1 + np.exp(-dataset))
def Softmax(dataset):
    exp_list = np.exp(dataset)
    return exp_list / np.sum(exp_list)

def Sigmoid_derivative(dataset):
    return Sigmoid(dataset) * (1- Sigmoid(dataset))