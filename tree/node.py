class DecisionTreeNode():
    '''
    Node class for Decision Tree.
    '''
    def __init__(self, X, Y, cri_fn):
        self.X = X
        self.Y = Y
        self.cri_fn = cri_fn
        self.loss = cri_fn(Y)
    def loss(self):
        return self.loss
    def get_criterion_fn(self):
        return self.cri_fn