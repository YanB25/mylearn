import math
import numpy as np
from logger import get_logger
import metrics
mylogger = get_logger(__name__ + '.MLPClassifier')

def hello():
    mylogger.debug('hello!')

class MLPClassifier():
    def __init__(
        self,
        hidden_layer_sizes=(20, ),
        activation='sigmoid',
        learning_rate_init=0.01,
        max_iter=5000,
        tol=1e-4,
        verbose=False,
        warm_start=True,
        random_stat=1,
        mini_batch='full'
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
        self.__last_loss = None
        self.mini_batch = mini_batch
        np.random.seed(random_stat)
        if not verbose:
            import logging
            mylogger.setLevel(logging.INFO)

        if self.activation == 'relu':
            self.activation = metrics.ReLu
        elif self.activation == 'leakyrelu':
            self.activation = metrics.LeakReLu
        elif self.activation == 'sigmoid':
            self.activate = metrics.Sigmoid
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

        if isinstance(self.mini_batch, str):
            if self.mini_batch == 'full' or self.mini_batch == 'not':
                self.mini_batch = self.n_samples_
            elif self.mini_batch == 'auto':
                self.mini_batch = math.floor(self.n_samples_/10)
        else:
            assert isinstance(self.mini_batch, int)

        val, cnt = np.unique(Y, return_counts=True)
        # self.n_output_ = val.shape[0]
        self.n_output_ = 1 #TODO: now output become 1
        self.n_classes_ = val

        # the i-th is Weight matrix map layer i to layer i+1
        # of shape (ls[i+1], ls[i])
        # 0 <= i <= n_layers-1
        self.coef_ = []

        # the i-th is bias vector at layer i
        # of shape (ls[i], n_sample_)
        # 0 <= i <= n_layers-2
        self.intercepts_ = []

        # TODO: output become 1 node!
        self.layer_sizes = (self.n_attr_, *self.hidden_layer_sizes, self.n_output_)
        self.n_layers_ = len(self.layer_sizes)

        self.__train()
    def __train(self):
        if self.warm_start:
            self.__warm_start()
        else:
            self.__cold_start()

        n_samples_ = self.mini_batch
        # index 0 means 1st layer Zs
        self.Zs = [
            np.zeros((self.layer_sizes[i+1], n_samples_))
            for i in range(self.n_layers_-1)
        ]

        # As = activate(Zs)
        other_As = [
            np.zeros((self.layer_sizes[i], n_samples_))
            for i in range(1, self.n_layers_)
        ]
        self.As = [self.X] + other_As

        # error is \frac{\dev Out}{\dev Zs}
        self.Error = [
            np.zeros((self.layer_sizes[i+1], n_samples_))
            for i in range(self.n_layers_ - 1)
        ]

        for i in range(self.n_layers_-1):
            mylogger.debug('init [%s] coef %s', i, self.coef_[i])
            mylogger.debug('init [%s] intercept %s', i, self.intercepts_[i])
            mylogger.debug('init [%s] Zs %s', i, self.Zs[i])
            mylogger.debug('init [%s] As %s', i, self.As[i])
            mylogger.debug('init [%s] Error %s', i, self.Error[i])
        mylogger.debug('init [%s] As %s', self.n_layers_-1, self.As[i])

        # start training here.
        for i in range(self.max_iter):

            # idx = np.random.choice(self.n_samples_, self.mini_batch)
            # X = self.X.iloc[:, idx]
            # Y = self.Y.iloc[idx, :]
            # mylogger.debug('MGK idx %s, X %s Y %s', idx,X.shape, Y.shape) 
            # self.As[0] = X #TODO: maybe bug

            loss = self.__feedforward(self.X, self.Y)
            self.__backpropagation(self.X, self.Y)
            mylogger.info('TRAINING: [%s] Loss %s', i, loss)
            if i % 100 == 0:
                mylogger.info('MG [%s] Loss %s', i, loss)
            if self.__last_loss is None:
                self.__last_loss = loss
            else:
                if loss > self.__last_loss:
                    mylogger.warn('[%s] loss %s, last loss %s. ERROR', i, loss, self.__last_loss)
                self.__last_loss = loss

    def __feedforward(self, X, Y):
        # 0 <= i_layer <= n_layers - 2
        for i_layer in range(self.n_layers_-1):
            WA = np.dot(self.coef_[i_layer], self.As[i_layer])
            WApB = WA + self.intercepts_[i_layer] 
            self.Zs[i_layer] = WApB
            self.As[i_layer + 1] = self.activate(self.Zs[i_layer])
            mylogger.debug("feedforward [l.%s] WA\n%s\nWApB\n%s\nZs\n%s\nAs\n%s",
                i_layer,
                WA,
                WApB,
                self.Zs[i_layer],
                self.As[i_layer + 1])
        outputA = self.As[self.n_layers_ - 1]
        mylogger.debug('Y shape %s, outputA shape %s', Y.shape, outputA.shape)
        loss = self.__loss(Y, outputA)
        mylogger.debug('feedforward loss is %s', loss)
        mylogger.debug('output A is %s', outputA)
        return loss

    def __backpropagation(self, X, Y):
        Y = np.array(Y).reshape((1, -1)) # WARNING: here, Y is 1 * n
        # for every layer, bp
        for i_layer in range(self.n_layers_ - 2, -1, -1):
            A = self.As[i_layer + 1]
            Z = self.Zs[i_layer]
            mylogger.debug('bp(%s) A %s, Z %s, layer-size %s', i_layer, A.shape, A.shape, self.layer_sizes[i_layer+1])
            assert A.shape[0] == Z.shape[0] == self.layer_sizes[i_layer + 1] # don't worry. python support that.
            assert A.shape[1] == Z.shape[1]

            # for the last layer

            # derivation of activation function.
            # TODO: WARNING: below only used in Sigmoid. please change me
            der = self.activate(Z) * (1- self.activate(Z))
            if i_layer == self.n_layers_ - 2:
                assert np.all(self.As[-1] == self.As[i_layer+1])
                assert np.all(self.Zs[-1] == self.Zs[i_layer])
                sm = -(Y / A) + (1-Y)/(1-A)
                assert sm.shape == A.shape
                assert der.shape == A.shape == Z.shape
                epsilon = sm * der
                assert epsilon.shape == A.shape
                mylogger.debug('bp(%s) Y\n%s\nA\n%s', i_layer, Y, A)
                mylogger.debug('bp(%s) left\n%s\nright\n%s', i_layer, -(Y/A), (1-Y)/(1-A))
                mylogger.debug('bp(%s) sm\n%s\nder\n%s', i_layer, sm, der)
                self.Error[i_layer] = epsilon

            else:
                W = self.coef_[i_layer + 1]
                E = self.Error[i_layer + 1]
                assert W.shape[1] == self.layer_sizes[i_layer + 1]
                assert W.shape[0] == self.layer_sizes[i_layer + 2]
                assert W.shape[0] == E.shape[0]
                mm = np.matmul(np.transpose(W), E)
                assert mm.shape == der.shape
                epsilon = np.multiply(mm, der)
                self.Error[i_layer] = epsilon

            mylogger.debug('bp(%s) : finished epsilon\n%s', i_layer, self.Error[i_layer])

            # now update W and B according to epsilon
            cur_n_size = self.layer_sizes[i_layer + 1]
            pre_n_size = self.layer_sizes[i_layer]
            epsilon = np.mean(self.Error[i_layer], axis=1)
            dCdB = epsilon.reshape((-1, 1))
            mylogger.debug('bp(%s) update. epsilon mean is \n%s\ndCdB is %s', i_layer, epsilon, dCdB)
            dCdW = np.zeros((cur_n_size, pre_n_size))

            udA = np.array(self.As[i_layer])
            epsilon = self.Error[i_layer]
            for idx in range(X.shape[1]):
                a = udA[:, idx].reshape((1, -1))
                eps = epsilon[:, idx].reshape((-1, 1))
                mylogger.debug('bp(%s) update. a shape %s dCdW shape %s eps shape %s', i_layer, a.shape, dCdW.shape, eps.shape)
                assert a.shape[1] == dCdW.shape[1]
                assert eps.shape[0] == dCdW.shape[0]
                w = eps * a
                mylogger.debug('bp(%s) [l.%s] update w(1/n) of %s', i_layer, idx, w)
                assert w.shape == dCdW.shape
                dCdW += w / X.shape[1]
            
            mylogger.debug('bp(%s) dCdB\n%s\ndCdW\n%s',i_layer, dCdB, dCdW)
            mylogger.debug('bp(%s) interf\n %s',i_layer, self.intercepts_[i_layer])
            # update
            assert self.intercepts_[i_layer].shape == dCdB.shape
            assert self.coef_[i_layer].shape == dCdW.shape
            self.intercepts_[i_layer] -= self.learning_rate_init * dCdB
            self.coef_[i_layer] -= self.learning_rate_init * dCdW

            mylogger.debug('bp(%s): update finish. B\n%s\nW\n%s',
                i_layer,
                self.intercepts_[i_layer],
                self.coef_[i_layer])


    def __loss(self, Y, A):
        '''
        calculate the typical loss in NN.
        @param Y :: np.array(Boolean), of shape (mini_batch, 1). ground truth.
        @param A :: pd.DataFrame(Float), of shape(n_output_, mini_batch). predicted truth.
        '''
        # ret = []
        # ai = np.array(ai)
        # y_i = np.array(yi).reshape(-1)
        # for i in range(self.n_output_):
        #     a_i = ai[i].reshape(-1)
        #     mylogger.debug('here %s %s', y_i.shape, a_i.shape)
        #     assert y_i.shape == a_i.shape
        #     ret.append(np.sum(np.where(yi == 1, -np.log(a_i), -np.log(1-a_i))))
        # return np.sum(ret)
        #TODO: loss is wrong for many output node.
        assert Y.shape[0] == A.shape[1]
        ret = []
        Y = Y.values.reshape(-1)
        mylogger.debug('cal loss begin. Y %s', Y)
        for i_node in range(A.shape[0]):
            ai = A[i_node, :]
            trans = np.where(Y == 1, -np.log(ai), -np.log(1-ai))
            mylogger.debug('cal loss(%s) ai %s trans %s', i_node, ai, trans)
            ret.append(np.sum(trans))

        return np.mean(ret)

    
    def __warm_start(self):
        self.coef_ = [
            np.random.random((self.layer_sizes[i+1], self.layer_sizes[i]))
            for i in range(self.n_layers_-1)
        ]
        self.intercepts_ = [
            np.random.random((self.layer_sizes[i+1], 1)) # WARNING: this has been an error. shape[1] should be 1
            for i in range(self.n_layers_-1)
        ]
    def __cold_start(self):
        self.coef_ = [
            np.zeros((self.layer_sizes[i+1], self.layer_sizes[i]))
            for i in range(self.n_layers_-1)
        ]
        self.intercepts_ = [
            np.zeros((self.layer_sizes[i+1], 1))
            for i in range(self.n_layers_-1)
        ]

    def predict(self, X):
        ret = []
        for idx in range(X.shape[1]):
            x = X.iloc[:, idx]
            ret.append(self.__predict(x))
        return ret
    def __predict(self, x):
        nx = x.values.reshape((-1, 1))
        for i_layer in range(self.n_layers_ - 1):
            wx = np.matmul(self.coef_[i_layer], nx)
            nx = wx + self.intercepts_[i_layer]
            nx = self.activate(nx)
        nx = nx.reshape(-1)[0]
        return (1 if nx > 0.5 else 0, nx)
    def score(self, X, Y):
        pass