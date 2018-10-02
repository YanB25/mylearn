'''
decision tree for classification and regression
'''
import pydot
import metrics
import numpy as np
from logger.logger import logger
from .node import DecisionTreeNode
import metrics
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
        criterion='id3', 
        max_depth=2, 
        delta_criterion_threshold=1e-1,
        min_batch=3
        ):
        self.__set_criterion(criterion)
        self.max_depth = min(max_depth, 5)
        self.delta_criterion_threshold = delta_criterion_threshold
        self.min_batch = min_batch
        self.__root = None
        self.__last_node_id = 0
    def __set_criterion(self, criterion):
        '''
        a helper function that used to set all criterion attr correct
        '''
        self.criterion = criterion
        if criterion == 'cart':
            self.criterion_fn = metrics.Gini
        elif criterion == 'id3':
            self.criterion_fn = metrics.Entropy
        elif criterion == 'c45':
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

        self.__root = Node(self.X, self.Y, self.criterion_fn, id=self.__last_node_id)
        self.__last_node_id += 1
        
        log_info = ''.join([
            'begin training decision tree classifier.\n',
            'criterion=%s\n',
            'max_depth=%s\n',
            'delta_criterion_threshold=%s\n',
            'min_batch=%s\n' 
        ])
        logger.info(
            log_info,
            self.criterion,
            self.max_depth,
            self.delta_criterion_threshold,
            self.min_batch)

        self.__train()

        logger.info('training finished.')

        assert self.__self_validate()
        logger.info('decision tree pass validation.')
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

        # pre-trunc: max_depth limit
        if depth >= self.max_depth:
            logger.debug('RET: reach max depth %s', depth)
            return
        # pre-trunc: loss is small enough
        if abs(node.loss()) <= epsilon:
            logger.debug('RET: loss %s less than epsilon', abs(node.loss()))
            return

        root_loss = node.loss()
        logger.debug('node loss %s', root_loss)

        logger.debug('column is %s', node.X.columns)

        # used to store and find the one get max objective
        objective_index = []
        for feature_i in node.X.columns:
            # below two line debug passed
            # no used. just comment out
            # for label, y in Y_groupby_feature_i:
            #     logger.debug('[%s], label: %s, y: %s', feature_i, label, y)

            # for entropy criterion
            if self.criterion == 'id3':
                delta_criterion = self.__entropy_gain_in_feature(node, feature_i)
                objective_index.append(delta_criterion)
            elif self.criterion == 'c45':
                delta_criterion = self.__entropy_rate_gain_in_feature(node, feature_i)
                objective_index.append(delta_criterion)
            elif self.criterion == 'cart':
                delta_criterion = self.__gini_coefficient_in_feature(node, feature_i)
                objective_index.append(delta_criterion)
            else:
                logger.error('ERROR unknown self.criterion %s', self.criterion)
                raise Exception('unknow tree self.criterion')
        
        # finished calculate all losses

        # if max objective is negative, no need to split anymore
        if np.max(np.abs(objective_index)) <= self.delta_criterion_threshold:
            logger.debug('RET: criterion gain %s less then threshold %s', np.max(np.abs(objective_index)), self.delta_criterion_threshold)
            return

        logger.debug('objective index is %s', objective_index)

        # get the best feature index
        max_feature_i = np.argmax(objective_index)
        node.by_feature_i = max_feature_i

        # now build child of the node
        best_feature = node.X.columns[max_feature_i]
        X_group_best = node.X.groupby(best_feature)
        logger.debug('best split by feature name %s', best_feature)

        # count how many row in dataframe
        count_row = lambda df: len(df.index)
        group_size = X_group_best.apply(count_row)
        logger.debug('each group size is %s', group_size)

        # pre-trunc: if least-size group is less than self.min_batch
        if np.min(group_size) < self.min_batch:
            logger.debug('RET: after split, min batch size %s less than self.min_batch %s limit.', np.min(group_size), self.min_batch)
            return

        for label, memb_idx in X_group_best.groups.items():
            logger.debug('label %s, group %s', label, memb_idx)
            logger.debug('candidate Xs are \n%s', node.X)

            # NOTICE: use loc instead of iloc.
            # in child node, iloc use absolute index, may raise out-of-bound index error.
            new_node = Node(
                node.X.loc[memb_idx], 
                node.Y.loc[memb_idx], 
                node.cri_fn, 
                acpt_feature=label,
                classes=node.classes,
                id=self.__last_node_id)
            self.__last_node_id += 1
            node.add_child(new_node)
            logger.debug('mount new node finished.')
            new_node.log_info()
            logger.debug('\n\n next recursive \n\n')
            self.__buildTree(new_node, depth + 1)
            
    def __entropy_gain_in_feature(self, node, feature_i):
        '''
        get how entropy changed when Y is group by feature_i
        input:
            - node :: Node, the node to be split and judged
            - feature_i :: str, the name of a feature title. (header of X's dataframe)
        output:
            - delta_criterion :: float, gain of entropy
        '''
        # in this function, tree must run under below two criterion
        assert self.criterion == 'id3' or self.criterion == 'c45'
        assert node.cri_fn == metrics.Entropy

        Y_groupby_feature_i = node.Y.groupby(node.X[feature_i])
        Y_group_criterion = Y_groupby_feature_i.apply(node.cri_fn)
        Y_group_cnt = Y_groupby_feature_i.apply(len)
        Y_group_prob = Y_group_cnt / np.sum(Y_group_cnt)
        logger.debug('[%s] loss: %s', feature_i, Y_group_criterion)
        logger.debug('[%s] Y: %s', feature_i, node.Y)
        logger.debug('[%s] cnt: %s', feature_i, Y_group_cnt)
        logger.debug('[%s] prob: %s', feature_i, Y_group_prob)

        target_criterion = np.sum(np.array(Y_group_prob) * np.array(Y_group_criterion))
        logger.debug('[%s] target_loss: %s', feature_i, target_criterion)

        # entropy gain
        # only below two variable are interact with outer side
        delta_criterion = node.loss() - target_criterion
        logger.debug('[%s]entropy gain(need max) is %s', feature_i, delta_criterion)
        return delta_criterion
    def __entropy_rate_gain_in_feature(self, node, feature_i):
        '''
        get how entropy RATE changed when Y is group by feature_i
        input:
            - node :: Node, the node to be split and judged
            - feature_i :: str, the name of a feature title. (header of X's dataframe)
        output:
            - delta_criterion_rate :: float, gain of entropy RATE.
                equal to __entropy_gain_in_feature(..) / entropy(feature[i])
        '''
        assert self.criterion == 'c45'
        assert node.cri_fn == metrics.Entropy

        entropy_gain = self.__entropy_gain_in_feature(node, feature_i)
        logger.debug('calculate attr-entropy at xs\n%s', self.X.iloc[:, feature_i])
        split_info_entropy = metrics.Entropy(self.X.iloc[:, feature_i])
        logger.debug('split entropy is %s', split_info_entropy)
        logger.debug('entropy gain rate is %s', entropy_gain / split_info_entropy)
        return entropy_gain / split_info_entropy
    def __gini_coefficient_in_feature(self, node, feature_i):
        '''
        calculate gini coefficient when spliting node by feature i.
        input:
            - node :: Node, the node to be split and judged
            - feature_i :: str, the name of a feature title. (header of X's dataframe)
        outout:
            - gini_coefficient :: float, the according coefficient
                NOTICE: less gini means better 
        '''
        assert self.criterion == 'cart'
        assert node.cri_fn == metrics.Gini

        Y_groupby_feature_i = node.Y.groupby(node.X[feature_i]) # node.X :: pd.DataFrame. feature_i is its feature name
        Y_group_gini = Y_groupby_feature_i.apply(node.cri_fn)
        Y_group_cnt = Y_groupby_feature_i.apply(len)
        Y_group_proba = Y_group_cnt / np.sum(Y_group_cnt)
        logger.debug('[%s] loss: %s', feature_i, Y_group_gini)
        logger.debug('[%s] X: %s', feature_i, node.X)
        logger.debug('[%s] Y: %s', feature_i, node.Y)
        logger.debug('[%s] cnt: %s', feature_i, Y_group_cnt)
        logger.debug('[%s] prob: %s', feature_i, Y_group_proba)

        node_feature_gini = np.sum(np.array(Y_group_gini) * np.array(Y_group_proba))
        logger.debug('[%s] (minus) node gini index: %s', feature_i, 0 - node_feature_gini)
        # because less gini means less uncertanty
        return 0 - node_feature_gini

    def __self_validate(self):
        '''
        after tree has been built, check whether some truth is broken.
        '''
        aft_sample = DecisionTreeClassifier.__calculate_samples(self.__root) 
        logger.info('validating tree. n_sample=%s, tree_sample=%s', self.n_sample, aft_sample)
        return aft_sample == self.n_sample

    @staticmethod
    def __calculate_samples(node):
        if node.children == []:
            logger.debug('leaf node has sample %s', node.Y.shape[0])
            return node.Y.shape[0]
        return np.sum([DecisionTreeClassifier.__calculate_samples(child) for child in node.children])
    def predict(self, predict_X):
        predict_Y = []
        for line in np.array(predict_X):
            logger.debug('predicting x_i %s label', line)
            predict_Y.append(self.__predict(line))
        return predict_Y
    def __predict(self, x):
        return self.__root.predict(x)
    def predict_prob(self, predict_X):
        predict_Y = []
        for line in np.array(predict_X):
            logger.debug('predicting x_i %s probability', line)
            predict_Y.append(self.__predict_prob(line))
        return predict_Y
    def __predict_prob(self, x):
        return self.__root.predict_prob(x)

    def depth(self):
        '''
        return depth of the tree
        output:
            depth :: int, the depth of the tree. root's depth is zero
        '''
        return self.__depth(self.__root)
    def __depth(self, node):
        if node.children == []:
            return 0
        return 1 + np.max([self.__depth(child) for child in node.children])
    def width(self):
        '''
        return num of leave in the tree
        output:
            width :: int, num of leave
        '''
        return self.__width(self.__root)
    def __width(self, node):
        if node.children == []:
            return 1
        return np.sum([self.__width(child) for child in node.children])

    def score(self, predict_X, predict_Y):
        '''
        get decision tree accuracy.
        input:
            - predict_X :: np.array of shape (n_sample, n_feature)
            - predict_Y :: np.array of shape (n_sample, )
        output:
            - socre :: float, correct-sample / n_sample
        '''
        # convert to np.array. Y should be of shape (x,)
        predict_X = np.array(predict_X)
        predict_Y = np.array(predict_Y).reshape(-1)

        self_predict_Y = self.predict(predict_X)
        same_list = self_predict_Y == predict_Y
        correct_n_sample = np.sum(same_list)

        n_sample = predict_Y.shape[0]
        logger.debug('ground truth Y %s, self-predict Y %s', predict_Y, self_predict_Y)
        logger.debug('compare result is %s', same_list)
        logger.debug('correct_n_sample %s, n_sample %s, score %s', correct_n_sample, n_sample, correct_n_sample/n_sample)

        logger.debug('error idx %s \nXs %s', np.where(~same_list), predict_X[np.where(~same_list)])
        return correct_n_sample/n_sample
    def build_graph(self, filename):
        '''
        build a graph of the tree. output as a png file
        input:
            - filename :: str, name of the png file, including 'png' extension
        '''
        if self.__root is None:
            logger.error('call DecisionTreeClassifier.fit before build_graph.')
            return
        self.graph = pydot.Dot(graph_type='graph')
        self.__build_graph(self.__root)
        self.graph.write_png(filename)
    def __build_graph(self, node):
        for child in node.children:
            edge = pydot.Edge(str(node), str(child))
            self.graph.add_edge(edge)
            self.__build_graph(child)
