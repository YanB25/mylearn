import numpy as np
import pandas as pd
import logging
from logger import logger
import math
def cross_val_score(clf, data, target, cv=10):
    '''
    perform k-fold cross validation
    input:
        - clf, classifier CLASS that implement interface `score`
        - data :: pd.DataFrame of shape (n_sample, n_feature)
        - target :: pd.DataFrame of shape (n_sample, )
        - cv :: Int, number of batch
    '''
    scores = []
    for i in range(cv):
        training_data, training_target, predicting_data, predicting_target = random_K_fold(data, target, cv)
        logger.debug('[cross %s]training_data %s', i, training_data)
        logger.debug('[cross %s]tarning_target %s', i, training_target)
        logger.debug('[cross %s]predicting_data %s', i, predicting_data)
        logger.debug('[cross %s]predicint_target %s', i, predicting_target)
        clf.fit(training_data, training_target)
        score = clf.score(predicting_data, predicting_target)
        scores.append(score)
    return scores
        


def unison_shuffled_copies(a, b, p):
    #assert len(a) == len(b)
    p = np.random.permutation(p)
    return a[p], b[p]

def random_K_fold(data, target, k):
    '''
    return K fold training and predicting set from data and target
    input:
        - data :: pd.DataFrame of shape(n_sample, n_attribute)
        - target :: pd.DataFrame of shape(n_samples)
        - k :: Int, the K to be fold
    output:
        - training_data :: pd.DataFrame of shape(n_sample - n_sample/k, n_attribute)
        - training_target :: pd.DataFrame of shape(n_sample - n_sample/k, )
        - predicting_data :: pd.DataFrame of shape(n_sample/k, n_attribute)
        - predicting_target :: pd.DataFrame of shape (n_sample/k)
    '''
    n_sample = target.shape[0]
    idxs = [i for i in range(n_sample)]
    logger.debug('[rkf] n_sample=%s', n_sample)
    np.random.shuffle(idxs)
    predict_len = math.floor(n_sample/k)
    logger.debug('[rkf] pindex %s, tindex %s', idxs[:predict_len], idxs[predict_len:])
    return data.loc[idxs[predict_len:] , :], target.loc[idxs[predict_len:] , :], data.loc[idxs[:predict_len], :], target.loc[idxs[:predict_len], :]