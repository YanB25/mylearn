import numpy as np
from logger import logger
class DecisionTreeNode():
    '''
    Node class for Decision Tree.
    '''
    def __init__(
        self, 
        X, 
        Y, 
        cri_fn, 
        classes=None, 
        acpt_feature=None):
        '''
        input:
            - X :: np.array of shape (n_sample, n_feature)
            - Y :: np.array of shape (n_sample, )
            - classes :: the predicted domain. MUST be the same in the whole tree
        '''
        self.X = X
        self.Y = Y
        self.n_sample = self.Y.shape[0]
        self.cri_fn = cri_fn
        self.__loss = cri_fn(Y)
        self.children = []
        self.by_feature_i = None
        self.acpt_feature = acpt_feature

        # for most occurency label y
        val, cnt = np.unique(self.Y, return_counts=True)
        max_idx = np.argmax(cnt)

        self.classes = val if classes is None else classes

        # a dict map from label to prob
        # for label occur in this node, calculate the fraction it occurs
        # for labels in tree but not in this node, return 0.
        nonezero_prob_dict = dict((label, count/self.n_sample) for label, count in zip(val, cnt))
        zero_dict = dict((label, 0) for label in self.classes if label not in val)

        # merge two dict
        self.prob_dict = {**nonezero_prob_dict, **zero_dict}
        logger.debug('node prob dict is %s', self.prob_dict)

        self.most_y = val[max_idx] # the label y that occurs most
        self.most_y_prob = cnt[max_idx] / Y.shape[0] # the prob that the y is correct(by prob)
        assert 0 <= self.most_y_prob <= 1 # do worry, python surpport that.
        assert isinstance(self.most_y_prob, float)
        logger.debug('node get most y %s for ys %s, at prob %s', self.most_y, self.Y, self.most_y_prob)

    def loss(self):
        return self.__loss
    def get_criterion_fn(self):
        '''
        return a criterion callable
        output:
            - self.cri_fn :: Callable([Iterable], [float])
        '''
        return self.cri_fn
    def add_child(self, new_node):
        '''
        append children node to the node
        '''
        self.children.append(new_node)
    def log_info(self):
        logger.debug('X is \n%s\n Y is \n%s\n, loss is %s, children are %s', self.X, self.Y, self.__loss, self.children)
    def __predict(self, x):
        '''
        recursively predict a sample x, return its predicted label and proba-list
        input:
            - x :: np.array((1, n_feature)), input sample
        output:
            - y :: Tuple(any, List(float)), the label predicted and the 
                proba of each classes, ordered as self.classes
        '''
        if self.children == []:
            proba_list = tuple(self.prob_dict[label] for label in self.classes)
            log_str = ''.join([
                'reach leaf. return most_y %s ',
                'and proba list %s\n',
                'classes are %s ',
            ])
            logger.debug(log_str, self.most_y, proba_list, self.classes)
            return self.most_y, proba_list
        comming_feature = x[self.by_feature_i]
        logger.debug('judge sample by feature index %s, get feature val %s', self.by_feature_i, comming_feature)
        for child in self.children:
            assert child.acpt_feature is not None
            if child.acpt_feature == comming_feature:
                return child.__predict(x)
        logger.error('error for sample x %s\n feature_i %s\nfeature value %s\n node acpt feature %s\n', x, self.by_feature_i, comming_feature, self.acpt_feature)
        raise Exception('Error! can not judge the sample!')

    def predict(self, x):
        '''
        return sample's label. label is of any type in self.classes
        input:
            - x :: np.array((1, n_feature))
        output:
            - label :: any, label of any type in self.classes
        '''
        return self.__predict(x)[0]
    def predict_prob(self, x):
        '''
        return sample's predicted probabilities list. proba are ordered as self.classes
        output:
            - proba-list :: [float] of length n_output
        '''
        return self.__predict(x)[1]


        