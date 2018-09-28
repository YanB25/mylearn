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
def Entropy(dataset):
    '''
    calculate entropy of a dataset
    input:
        - dataset :: Iterable
    output:
        - entropy :: float
    '''
    val, cnt = np.unique(dataset, return_counts=True)
    cnt_prob = cnt / np.sum(cnt)
    cnt_pro_logpro = cnt_prob * np.log2(cnt_prob)
    return 0 - np.sum(cnt_pro_logpro)

metrics_fn = {
    'gini': Gini,
    'entropy': Entropy
}