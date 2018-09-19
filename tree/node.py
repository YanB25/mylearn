class DecisionTreeNode():
    '''
    Node class for Decision Tree.
    '''
    def __init__(self, X, Y, fn='gini'):
        self.X = X
        self.Y = Y
        self.fn_name = fn
        pass
    