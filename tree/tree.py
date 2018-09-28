import metrics
'''
decision tree for classification and regression
'''
from logger.logger import logger
from .node import DecisionTreeNode

# renaming
Node = DecisionTreeNode

def hello():
    logger.debug('hello!!!')

class DecisionTreeClassifier():
    '''
    decision tree used for classification
    '''
    def __init__(self, criterion='entropy', max_depth=None):
        self.__set_criterion()
        self.criterion = metrics_fn[criterion]
        self.max_depth = max_depth
        self.__root = None
    def __set_criterion(self):
        self.criterion = criterion
        if criterion == 'gini':
            self.criterion_fn = metrics.Gini
        elif criterion == 'entropy':
            self.criterion_fn = metrics.Entropy
        else:
            raise Exception("para name unkonw {}".format(criterion))
    def fit(X, Y):
        '''
        fit and train the tree
        input:
            - X: pd.DataFrame[n_samples, n_features]
            - Y: pd.DataFrame[n_samples, 1]
        '''
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n_sample, self.n_feature = self.X.shape
        logger.debug('sample %s, feature %s', self.n_sample, self.n_feature)

        self.__root = Node(self.X, self.Y, self.criterion_fn)
        
        self.__train()
    def __train(self):
        self.__buildTree(self.__root)
    def __buildTree(self, node):
        root_loss = self.__root.loss()
        logger.debug('root loss {}', root_loss)
        for feature_idx in range(self.n_feature):


    def predict(self):
        pass
    def predict_prob(self):
        pass
    def score(self):
        pass
    