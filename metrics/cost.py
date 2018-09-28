import numpy as np
from logger import logger
def Gini(dataset):
    '''
    calculate gini coefficient of a dataset
    input:
        - dataset :: Iterable
    output:
        - gini_coef :: float
    '''
    val, cnt = np.unique(dataset, return_counts=True)
    cnt_prob = cnt / np.sum(cnt)
    return 1 - np.sum(cnt_prob ** 2)