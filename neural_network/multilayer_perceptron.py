import numpy as np
from logger import get_logger
import metrics
mylogger = get_logger(__name__)

def hello():
    mylogger.debug('hello!')

class MLPClassifier():
    def __init__(
        self,
        hidden_layer_sizes=(20, ),
        activation='relu',
        learning_rate_init=0.1,
        max_iter=5000,
        tol=1e-4,
        verbose=False,
        warm_start=True
    ):
        '''
        @param hidden_layer_sides :: Tuple(Int). the hidden layer sizes.
        @param activation :: str, in ['relu']. Denote the activation function
        @param learning_rate_init :: Float, the learning rate.
        @param max_iter :: Int, the maximun interation. Stop Iteration after
            that.
        @param tol :: tolorance of optimations.
        @param verbose :: Boolean. whether enable debug.
        @param warm_start :: Boolean. whether warm start.
        '''
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        if not verbose:
            import logging
            mylogger.setLevel(logging.WARNING)

        if self.activation == 'relu':
            self.activate = metrics.ReLu
        elif self.activate == 'leakyrelu':
            self.activate = metrics.LeakReLu
        else:
            raise NotImplementedError('unknown activation {}'.format(self.activation))
    def fit(self, X, Y):
        '''
        @param X :: pd.DataFrame(Float). of shape (n_attr_, n_samples_)
        @param Y :: pd.DataFrame(Float). of shape (n_attr_, )
        '''
        self.X = X
        self.Y = Y

        self.n_samples_ = X.shape[1]
        self.n_attr_ = X.shape[0]

        val, cnt = np.unique(Y, return_counts=True)
        self.n_output_ = val.shape[0]
        self.n_classes_ = val

        # the i-th is Weight matrix at layer (i-1)
        # of shape (ls[i+1], ls[i])
        # 0 <= i <= n_layers-1
        self.coef_ = []

        # the i-th is bias vector at layer (i-1)
        # of shape (ls[i+1], n_sample)
        # 0 <= i <= n_layers-1
        self.intercepts_ = []

        self.layer_sizes = (self.n_attr_, *self.hidden_layer_sizes, self.n_output_)
        self.n_layers_ = len(self.layer_sizes)

        self.__train()
    def __train(self):
        if self.warm_start:
            self.__warm_start()
        else:
            self.__cold_start()

        # index 0 means 1st layer Zs
        self.Zs = [
            np.zeros((self.layer_sizes[i+1], ))
            for i in range(self.n_layers_-1)
        ]

        # As = activate(Zs)
        self.As = [
            np.zeros((self.layer_sizes[i+1], ))
            for i in range(self.n_layers_-1)
        ]

        # error is \frac{\dev Out}{\dev Zs}
        self.Error = [
            np.zeros((self.layer_sizes[i+1], ))
            for i in range(self.n_layers_ - 1)
        ]

        for i in range(self.n_layers_-1):
            mylogger.debug('init [%s] coef %s', i, self.coef_[i])
            mylogger.debug('init [%s] intercept %s', i, self.intercepts_[i])
            mylogger.debug('init [%s] Zs %s', i, self.Zs[i])
            mylogger.debug('init [%s] As %s', i, self.As[i])
            mylogger.debug('init [%s] Error %s', i, self.Error[i])
        self.__feedforward(self.X, self.Y)
        self.__backpropagation(self.X, self.Y)
    def __feedforward(self, X, Y):
        # 0 <= i_layer <= n_layers - 2
        for i_layer in range(self.n_layers_-1):
            # the first one
            if i_layer == 0:
                # Z(i) = W(i) (mat)* X + B(i)
                # A(i) = Sigma(Z(i))
                WX = np.dot(self.coef_[i_layer], X)
                WXpB = WX + self.intercepts_[i_layer]
                self.Zs[i_layer] = WXpB

                mylogger.debug('ff shape [%s]: coef %s, inter %s, X %s, Z %s, A %s', 
                    i_layer, 
                    self.coef_[i_layer].shape, 
                    self.intercepts_[i_layer].shape, 
                    X.shape,
                    self.Zs[i_layer].shape,
                    '=X\'s'
                )
                mylogger.debug('ff val [%s]: WX %s\nWX + B %s\nZ %s\n', 
                    i_layer, 
                    WX,
                    WXpB,
                    self.Zs[i_layer]
                )
            else:
                # Z(i) = W(i) (mat)* A(i-1) + B(i)
                # A(i) = Sigma(Z(i))
                WA = np.dot(self.coef_[i_layer], self.As[i_layer-1])
                WApB = WA + self.intercepts_[i_layer]
                self.Zs[i_layer] = WApB

                mylogger.debug('ff shape [%s]: coef %s, inter %s, Z %s, A %s', 
                    i_layer,
                    self.coef_[i_layer].shape,
                    self.intercepts_[i_layer].shape,
                    self.Zs[i_layer].shape,
                    self.As[i_layer-1].shape,
                    )

                mylogger.debug('ff val [%s]: WA %s\nWApB %s\nZs %s',
                    i_layer,
                    WA,
                    WApB,
                    self.Zs[i_layer]
                )

            self.As[i_layer] = self.activate(self.Zs[i_layer])
            mylogger.debug('finish layer [%s], Zs %s As %s', i_layer, self.Zs[i_layer].shape, self.As[i_layer].shape)
            mylogger.debug('finish layer [%s], \nZs %s\nAs %s', i_layer, self.Zs[i_layer], self.As[i_layer])
        
        # reach last layer. now predict
        # get the first positive A's proba
        predict = np.transpose(self.As[-1][0])
        mylogger.debug('predict is %s', predict)
        predict_softmax = np.exp(predict) / np.sum(np.exp(predict))
        mylogger.debug('softmax is %s', predict_softmax)
        loss = MLPClassifier.__loss(Y, predict_softmax)
        mylogger.debug('loss is %s', loss)
        return loss

    def __backpropagation(self, X, Y):
        X = np.array(X)
        Y = np.array(Y).reshape(-1)
        # n_layers_ >= i_layer >= 0
        for i_layer in range(self.n_layers_ - 2, -1, -1):
            # if is last layer
            if i_layer == self.n_layers_ - 2:
                left= -(Y / self.As[i_layer])
                right = (1-Y)/(1-self.As[i_layer])
                self.Error[-1] = left + right
                
                mylogger.debug('bp [%s] Y %s, As %s, Error %s, <left> %s, <right> %s',
                    i_layer,
                    Y.shape,
                    self.As[i_layer].shape,
                    self.Error[-1].shape,
                    left.shape,
                    right.shape
                )
                mylogger.debug('bp [%s] Y %s\nAs %s\nError %s\nleft %s\nright %s',
                    i_layer,
                    Y,
                    self.As[i_layer],
                    self.Error[-1],
                    left,
                    right)
            else:
                WError = np.dot(np.transpose(self.coef_[i_layer+1]), self.Error[i_layer+1])
                Error = WError * np.where(self.Zs[i_layer] > 0, self.Zs[i_layer], 0)
                self.Error[i_layer] = Error
                mylogger.debug('bp [%s], error is %s, coef is %s', i_layer, self.Error[i_layer+1].shape, self.coef_[i_layer+1].shape)
                mylogger.debug('bp [%s], coef %s, Error(i+1) %s, WError %s, Error(i) %s',
                    i_layer,
                    self.coef_[i_layer+1],
                    self.Error[i_layer+1],
                    WError,
                    self.Error[i_layer])




            
    @staticmethod
    def __loss(yi, ai):
        '''
        calculate the typical loss in NN.
        @param yi :: Boolean, ground truth.
        @param ai :: Float, predicted truth.
        '''
        return np.sum(np.where(yi == 1, -np.log(ai), -np.log(1-ai)))


    def activate(self, mat):
        pass

                


    
    def __warm_start(self):
        self.coef_ = [
            np.random.random((self.layer_sizes[i+1], self.layer_sizes[i]))
            for i in range(self.n_layers_-1)
        ]
        self.intercepts_ = [
            np.random.random((self.layer_sizes[i+1], self.n_samples_))
            for i in range(self.n_layers_-1)
        ]
    def __cold_start(self):
        self.coef_ = [
            np.zeros((self.layer_sizes[i+1], self.layer_sizes[i]))
            for i in range(self.n_layers_-1)
        ]
        self.intercepts_ = [
            np.zeros((self.layer_sizes[i+1], self.n_samples_))
            for i in range(self.n_layers_-1)
        ]

    def predict(self, X):
        pass
    def score(self, X, Y):
        pass