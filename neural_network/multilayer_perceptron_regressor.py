import os
import pickle
import math
import numpy as np
from logger import get_logger
import metrics
mylogger = get_logger(__name__ + '.MLPClassifier')

def hello():
    mylogger.debug('hello!')

class MLPRegressor():
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
        mini_batch='full',
        step_size=100,
        load_from_file=False,
        dump_file=False,
        file_root='save-data',
        validation_set = None
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
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.__last_loss = None
        self.mini_batch = mini_batch
        self.step_size = step_size
        self.load_from_file = load_from_file
        self.beg_index = 0
        self.dump_file = dump_file
        self.loss_gain_cnt = 0
        self.file_root = file_root
        self.no_dataset_shuffle = mini_batch == 'not'
        self.information = {}
        if validation_set:
            self.validation_X, self.validation_Y = validation_set
        else:
            self.validation_X, self.validation_Y = (None, None)
        np.random.seed(random_stat)
        if not verbose:
            import logging
            mylogger.setLevel(logging.INFO)

        if self.activation == 'relu':
            self.activate = metrics.ReLu
            self.activate_derivative = metrics.ReLu_derivative
            self.last_activate = metrics.ReLu
            self.last_activate_derivative = metrics.ReLu_derivative
        elif self.activation == 'leakyrelu':
            self.activate = metrics.LeakReLu
            self.activate_derivative = metrics.LeakReLu_derivative
        elif self.activation == 'sigmoid':
            self.activate = metrics.Sigmoid
            self.activate_derivative = metrics.Sigmoid_derivative
            self.last_activate = metrics.Sigmoid
            self.last_activate_derivative = metrics.Sigmoid_derivative
        elif self.activation == 'id':
            self.activate = metrics.Id
            self.activate_derivative = metrics.Id_derivative
            self.last_activate = metrics.Id
            self.last_activate_derivative = metrics.Id_derivative

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

        self.n_output_ = 1

        # the i-th is Weight matrix map layer i to layer i+1
        # of shape (ls[i+1], ls[i])
        # 0 <= i <= n_layers-1
        self.coef_ = []

        # the i-th is bias vector at layer i
        # of shape (ls[i], n_sample_)
        # 0 <= i <= n_layers-2
        self.intercepts_ = []

        self.layer_sizes = (self.n_attr_, *self.hidden_layer_sizes, self.n_output_)
        self.n_layers_ = len(self.layer_sizes)


        self.__init_data()
        if self.load_from_file:
            self.__init_from_file()

        self.__train()
    def __init_data(self):
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
    def __init_from_file(self):
        for idx in range(self.max_iter, -1, -1):
            name = '{}/{}-{}.cachedata'.format(self.file_root, 'coef', str(idx))
            if os.path.isfile(name):
                self.beg_index = idx
                f = open(name, 'rb')
                self.coef_ = pickle.load(f)
                mylogger.info('reload from file %s', name)

                name = '{}/{}-{}.cachedata'.format(self.file_root, 'inter', str(idx))
                assert os.path.isfile(name)
                f = open(name, 'rb')
                self.intercepts_ = pickle.load(f)
                mylogger.info('reload from file %s', name)


                name = '{}/{}-{}.cachedata'.format(self.file_root, 'information', str(idx))
                assert os.path.isfile(name)
                f = open(name, 'rb')
                self.information= pickle.load(f)
                mylogger.info('reload from file %s', name)

                return

    def __train(self):
        # start training here.
        for i in range(self.beg_index, self.max_iter):

            if self.no_dataset_shuffle:
                X = self.X
                Y = self.Y
            else:
                idx = np.random.choice(self.n_samples_, self.mini_batch)
                X = self.X.iloc[:, idx]
                Y = self.Y.iloc[idx, :]
            self.As[0] = X

            # pickle
            if self.dump_file and i % self.step_size == 0:
                import pickle
                name = '{}/{}-{}.cachedata'.format(self.file_root, 'coef', str(i))
                f = open(name, 'wb')
                pickle.dump(self.coef_, f)
                f.close()
                name = '{}/{}-{}.cachedata'.format(self.file_root, 'inter', str(i))
                f = open(name, 'wb')
                pickle.dump(self.intercepts_, f)
                name = '{}/{}-{}.cachedata'.format(self.file_root, 'information', str(i))
                f = open(name, 'wb')
                pickle.dump(self.information, f)

            if i % self.step_size == 0:
                loss = self.__feedforward(X, Y, return_loss=True)
                mylogger.info('[%s] loss %s', i, loss)
            else:
                self.__feedforward(X, Y, return_loss=False)
            self.__backpropagation(X, Y)

            if i % self.step_size == 0:
                if self.mini_batch == 'not' and self.__last_loss and loss > self.__last_loss:
                    self.learning_rate /= 2
                    mylogger.info('decrease learning rate to %s', self.learning_rate)
                if self.__last_loss is None:
                    self.__last_loss = loss
                else:
                    self.__last_loss = loss

                # and then validation
                if self.validation_X is not None:
                    s = self.score(self.validation_X, self.validation_Y)
                    s2 = self.score(X, Y)
                    mylogger.info('validation score %s, training score %s', s, s2)
                    self.information[i] = (loss, s, s2)
                else:
                    self.information[i] = loss

    def __feedforward(self, X, Y, return_loss=False):
        # 0 <= i_layer <= n_layers - 2
        for i_layer in range(self.n_layers_-1):
            WA = np.dot(self.coef_[i_layer], self.As[i_layer])
            WApB = WA + self.intercepts_[i_layer] 
            self.Zs[i_layer] = WApB
            if i_layer == self.n_layers_ - 2:
                self.As[i_layer + 1] = self.last_activate(self.Zs[i_layer])
            else:
                self.As[i_layer + 1] = self.activate(self.Zs[i_layer])
            mylogger.debug("feedforward [l.%s] WA\n%s\nWApB\n%s\nZs\n%s\nAs\n%s",
                i_layer,
                WA,
                WApB,
                self.Zs[i_layer],
                self.As[i_layer + 1])
        outputA = self.As[self.n_layers_ - 1]
        mylogger.debug('Y shape %s, outputA shape %s', Y.shape, outputA.shape)
        mylogger.debug('output A is %s', outputA)
        if return_loss:
            loss = self.__loss(Y, outputA)
            mylogger.debug('feedforward loss is %s', loss)
            return loss / Y.shape[0]

    def __backpropagation(self, X, Y):
        Y = np.array(Y).reshape((1, -1)) # WARNING: here, Y is 1 * n
        # for every layer, bp
        for i_layer in range(self.n_layers_ - 2, -1, -1):
            A = self.As[i_layer + 1]
            Z = self.Zs[i_layer]
            mylogger.debug('bp(%s) A %s, Z %s, layer-size %s', i_layer, A.shape, A.shape, self.layer_sizes[i_layer+1])
            assert A.shape[0] == Z.shape[0] == self.layer_sizes[i_layer + 1] # don't worry. python support that.
            assert A.shape[1] == Z.shape[1]


            # derivation of activation function.
            # TODO: WARNING: below only used in Sigmoid. please change me
            # TODO: I am trying to change. has it bug?

            # for the last layer
            if i_layer == self.n_layers_ - 2:
                der = self.last_activate_derivative(Z)
                assert np.all(self.As[-1] == self.As[i_layer+1])
                assert np.all(self.Zs[-1] == self.Zs[i_layer])
                sm = self.__loss_derivative(Y, A)
                assert sm.shape == A.shape
                assert der.shape == A.shape == Z.shape
                epsilon = sm * der
                assert epsilon.shape == A.shape
                mylogger.debug('bp(%s) Y\n%s\nA\n%s', i_layer, Y, A)
                mylogger.debug('bp(%s) left\n%s\nright\n%s', i_layer, -(Y/(A+1e-3)), (1-Y)/(1-A-1e-3))
                mylogger.debug('bp(%s) sm\n%s\nder\n%s', i_layer, sm, der)
                self.Error[i_layer] = epsilon

            else:
                der = self.activate_derivative(Z)
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
            self.intercepts_[i_layer] -= self.learning_rate * dCdB
            self.coef_[i_layer] -= self.learning_rate * dCdW

            mylogger.debug('bp(%s): update finish. B\n%s\nW\n%s',
                i_layer,
                self.intercepts_[i_layer],
                self.coef_[i_layer])


    def __loss(self, Y, A):
        '''
        calculate the typical loss in NN.
        @param Y :: np.array(Boolean), of shape (mini_batch, 1). ground truth.
        @param A :: pd.DataFrame(Float), of shape(1, mini_batch). predicted truth.
        '''
        # assert Y.shape[0] == A.shape[1]
        # ret = []
        # Y = Y.values.reshape(-1)
        # mylogger.debug('cal loss begin. Y %s', Y)
        # for i_node in range(A.shape[0]):
        #     ai = A[i_node, :]
        #     ai = np.where(ai > 1-1e-5, 1-1e-5, ai)
        #     ai = np.where(ai < 1e-5, 1e-5, ai)
        #     trans = np.where(Y == i_node, -np.log(ai + 1e-6), -np.log(1-ai-1e-6))
        #     mylogger.debug('cal loss(%s) ai %s trans %s', i_node, ai, trans)
        #     ret.append(np.sum(trans))

        # return np.mean(ret)
        assert Y.shape[1] == 1
        assert A.shape[0] == 1
        Y = Y.values.reshape(-1)
        A = np.array(A).reshape(-1)
        assert Y.shape[0] == A.shape[0]
        dist = np.sum((Y-A)**2)
        return math.sqrt(dist)

    def __loss_derivative(self, Y, A):
        Y = Y.reshape((1, -1))
        assert Y.shape == A.shape
        return 2 * (A - Y)
    def last_activate(self, dataset):
        pass
    def last_activate_derivative(self, dataset):
        pass
    
    def __warm_start(self):
        self.coef_ = [
            np.random.random((self.layer_sizes[i+1], self.layer_sizes[i])) * 0.5
            for i in range(self.n_layers_-1)
        ]
        self.intercepts_ = [
            np.random.random((self.layer_sizes[i+1], 1))  * 0.5 # WARNING: this has been an error. shape[1] should be 1
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
        '''
        predict the y label of X
        @param X :: pd.DataFrame, of shape(n_attr_, n_samples_)
        @ret ret :: [(label, output_a)]
        '''
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
            #nx = self.activate(nx)
            if i_layer == self.n_layers_ - 2:
                nx = self.last_activate(nx)
            else:
                nx = self.activate(nx)
        nx = nx.reshape(-1)[0]
        return nx

    def score(self, X, Y):
        '''
        predict the y label of X
        @param X :: pd.DataFrame, of shape(n_attr_, n_samples_)
        @param Y :: pd.DataFrame, of shape(n_samples, 1)
        @ret ret :: Float, accuracy
        '''
        # p = self.predict(X)
        # error = 0
        # ys = Y.values.reshape(-1)
        # n_samples = 0
        # for i in zip(p, ys):
        #     n_samples += 1
        #     left, truth = i
        #     error += abs(left - truth)
        # return error/ys.shape[0]
        p = self.predict(X)
        p = np.array(p).reshape(-1)
        Y = Y.values.reshape(-1)
        result = np.corrcoef(p, Y)
        return result[0][1]

    def to_csv(self, test_X, filename):
        '''
        为了完成作业而开的接口。
        @param test_Y :: pd.DataFrame, of shape(n_attr, n_sample)
        @param filename :: Str, the filename
        '''
        p = self.predict(test_X)
        result = pd.DataFrame([i[0] for i in p])
        result.to_csv(filename, index=0, header=None, index_label=None)
