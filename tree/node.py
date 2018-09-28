from logger import logger
class DecisionTreeNode():
    '''
    Node class for Decision Tree.
    '''
    def __init__(self, X, Y, cri_fn):
        #TODO: add feature_i and feature. so that in predict step, it can be referenced
        self.X = X
        self.Y = Y
        self.cri_fn = cri_fn
        self.__loss = cri_fn(Y)
        self.children = []
    def loss(self):
        return self.__loss
    def get_criterion_fn(self):
        return self.cri_fn
    def add_child(self, new_node):
        self.children.append(new_node)
    def log_info(self):
        logger.info('X is \n%s\n Y is \n%s\n, loss is %s, children are %s', self.X, self.Y, self.__loss, self.children)