import numpy as np
from logger import logger
class DecisionTreeNode():
    '''
    Node class for Decision Tree.
    '''
    def __init__(self, X, Y, cri_fn, acpt_feature=None):
        #TODO: add feature_i and feature. so that in predict step, it can be referenced
        self.X = X
        self.Y = Y
        self.cri_fn = cri_fn
        self.__loss = cri_fn(Y)
        self.children = []
        self.by_feature_i = None
        self.acpt_feature = acpt_feature

        # for most occurency label y
        val, cnt = np.unique(self.Y, return_counts=True)
        max_idx = np.argmax(cnt)
        self.most_y = val[max_idx]
        logger.debug('node get most y %s for ys %s', self.most_y, self.Y)
    def loss(self):
        return self.__loss
    def get_criterion_fn(self):
        return self.cri_fn
    def add_child(self, new_node):
        self.children.append(new_node)
    def log_info(self):
        logger.info('X is \n%s\n Y is \n%s\n, loss is %s, children are %s', self.X, self.Y, self.__loss, self.children)
    def predict(self, x):
        '''
        recursively predict a sample x
        input:
            - x :: np.array((1, n_feature)), input sample
        output:
            - y :: any, the label predicted
        '''
        if self.children == []:
            logger.debug('reach leaf. return most_y %s', self.most_y)
            return self.most_y
        comming_feature = x[self.by_feature_i]
        logger.debug('judge sample by feature index %s, get feature val %s', self.by_feature_i, comming_feature)
        for child in self.children:
            assert child.acpt_feature is not None
            if child.acpt_feature == comming_feature:
                return child.predict(x)
        logger.error('error for sample x %s\n feature_i %s\nfeature value %s\n node acpt feature %s\n', x, self.by_feature_i, comming_feature, self.acpt_feature)
        raise Exception('Error! can not judge the sample!')
         
        