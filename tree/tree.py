import metrics
import numpy as np
'''
decision tree for classification and regression
'''
from logger.logger import logger
from .node import DecisionTreeNode

# renaming a class
Node = DecisionTreeNode

# a  very small number
epsilon = 1e-3

def hello():
    logger.debug('hello!!!')

class DecisionTreeClassifier():
    '''
    decision tree used for classification
    '''
    def __init__(
        self, 
        criterion='entropy', 
        max_depth=2, 
        delta_criterion_threshold=1e-1,
        min_batch=2
        ):
        self.__set_criterion(criterion)
        self.max_depth = min(max_depth, 5)
        self.delta_criterion_threshold = delta_criterion_threshold
        self.min_batch = min_batch
        self.__root = None
    def __set_criterion(self, criterion):
        self.criterion = criterion
        if criterion == 'gini':
            self.criterion_fn = metrics.Gini
        elif criterion == 'entropy':
            self.criterion_fn = metrics.Entropy
        else:
            raise Exception("para name unkonw {}".format(criterion))
    def fit(self, X, Y):
        '''
        fit and train the tree
        input:
            - X: pd.DataFrame[n_samples, n_features]
            - Y: pd.DataFrame[n_samples, 1]
        '''
        self.X = X
        self.Y = Y
        self.n_sample, self.n_feature = self.X.shape
        logger.debug('sample %s, feature %s', self.n_sample, self.n_feature)

        self.__root = Node(self.X, self.Y, self.criterion_fn)
        
        self.__train()
    def __train(self):
        self.__buildTree(self.__root, 0)
    def __buildTree(self, node, depth):
        '''
        build the whole decision tree.
        input:
            - node :: Node, a recursive function, input a node object, try to find
                and build it's children
            - depth :: Int, zero-base, calculate the current depth
        '''
        # all the node's criterion function MUST be the same 
        # as the tree
        assert self.criterion_fn == node.cri_fn

        #TODO: recuisive base here
        if depth >= self.max_depth:
            logger.debug('RET: reach max depth %s', depth)
            return
        if abs(node.loss()) <= epsilon:
            logger.debug('RET: loss %s less than epsilon', abs(node.loss()))
            return
        # if dataset at node is less than min_batch
        batch_size = np.array(node.X).reshape((1, -1)).shape[-1]
        if batch_size <= self.min_batch:
            logger.info('RET: batch size %s reach min_batch %s', batch_size, self.min_batch)
            return


        root_loss = node.loss()
        logger.debug('node loss %s', root_loss)

        logger.debug('column is %s', node.X.columns)

        # used to store and find the one get max objective
        objective_index = []
        for feature_i in node.X.columns:
            Y_groupby_feature_i = node.Y.groupby(node.X[feature_i])

            # below two line debug passed
            # no used. just comment out
            for label, y in Y_groupby_feature_i:
                logger.debug('[%s], label: %s, y: %s', feature_i, label, y)

            # for entropy criterion
            if self.criterion == 'entropy':
                Y_group_criterion = Y_groupby_feature_i.apply(node.cri_fn)
                Y_group_cnt = Y_groupby_feature_i.apply(len)
                Y_group_prob = Y_group_cnt / np.sum(Y_group_cnt)
                logger.debug('[%s] loss: %s', feature_i, Y_group_criterion)
                logger.debug('[%s] cnt: %s', feature_i, Y_group_cnt)
                logger.debug('[%s] prob: %s', feature_i, Y_group_prob)

                target_criterion = np.sum(np.array(Y_group_prob) * np.array(Y_group_criterion))
                logger.debug('[%s] target_loss: %s', feature_i, target_criterion)

                # entropy gain
                # only below two variable are interact with outer side
                delta_criterion = node.loss() - target_criterion
                logger.debug('[%s]entropy gain(need max) is %s', feature_i, delta_criterion)

                objective_index.append(delta_criterion)
        
        # finished calculate all losses

        # if max objective is negative, no need to split anymore
        if max(objective_index) <= self.delta_criterion_threshold:
            logger.debug('RET: criterion gain %s less then threshold %s', max(objective_index), self.delta_criterion_threshold)
            return

        logger.debug('objective index is %s', objective_index)
        max_feature_i = np.argmax(objective_index)

        # now build child of the node
        best_feature = node.X.columns[max_feature_i]
        X_group_best = node.X.groupby(best_feature)
        for label, memb_idx in X_group_best.groups.items():
            logger.debug('label %s, group %s', label, memb_idx)
            logger.debug('candidate Xs are \n%s', node.X)

            # NOTICE: use loc instead of iloc.
            # in child node, iloc use absolute index, may raise out-of-bound index error.
            new_node = Node(node.X.loc[memb_idx], node.Y.loc[memb_idx], node.cri_fn)
            node.add_child(new_node)
            logger.debug('mount new node finished.')
            new_node.log_info()
            logger.debug('\n\n next recursive \n\n')
            self.__buildTree(new_node, depth + 1)
            
    def predict(self):
        pass
    def predict_prob(self):
        pass
    def score(self):
        pass
    