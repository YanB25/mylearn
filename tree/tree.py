'''
decision tree for classification and regression
'''
from logger.logger import logger
def hello():
    logger.debug('hello!!!')
class DecisionTreeClassifier():
    '''
    decision tree used for classification
    '''
    def __init__(self, criterion, max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        pass
    def fit(X, Y):
        pass
    def predict():
        pass
    def predict_prob():
        pass
    def score():
        pass
    