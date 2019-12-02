import numpy as np
from random import randrange
from random import sample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

style.use("ggplot")
import sys


class FCN:
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    def network_topology(self, K):
        self.G1 = np.array([[1, 0, 1, 0, 0],
                            [0, 1, 0, 1, 1],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]])
        self.G1 = self.G1 + self.G1.T - np.eye(K)
        self.G = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if self.G1[i, j] == 1:
                    if i == j:
                        pass
                    else:
                        self.G[i, j] = 1 / (max(sum(self.G1[i, :]) - 1, sum(self.G1[j, :]) - 1) + 1)
                if j == K - 1:
                    self.G[i, i] = 1 - sum(self.G[i, :])
        print(self.G.round(decimals=3))
        pass

    def progress(self, i):
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * round(i / 10), (100 / self.num_epochs) * i))
        sys.stdout.flush()

    def affine_forward(self, x, w, b):
        z = None
        X = x
        z = X @ w + b  # z = WX + b
        cache = (x, w, b)
        return z, cache

    def affine_backward(self, dout, cache):
        x, w, b = cache
        dx, dw, db = None, None, None
        N = x.shape[0]
        X = np.reshape(x, (N, -1))  # dout=dLoss/dz - gradient from previous step
        dx = dout @ w.T   # dz/dx = dot(W.T, dout)
        dw = X.T @ dout  # dz/wd = dot(dout, X)
        db = np.sum(dout, axis=0)   # dz/db = sum(dout)
        return dx, dw, db

    def relu_forward(self, x):
        cache = x
        out = x * (x > 0)  # ReLu
        return out, cache

    def relu_backward(self, dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx

    def softmax_loss(self, x, y):
        N = x.shape[0]
        shiftx = x.T - x.T.max(0)
        shiftx = np.exp(shiftx)
        out = shiftx / np.sum(shiftx, axis=0)
        out = out.T
        loss = np.log(out[(np.arange(N), y)] / np.sum(out, axis=1) + self.epsilon)
        loss = -np.sum(loss) / N
        dout = out.copy()# dLoss/dx
        dout[(np.arange(N), y)] -= 1
        dout /= N
        return loss, dout

    def affine_relu_forward(self, x, w, b):
        z_fc, cache_fc = self.affine_forward(x, w, b)
        z_relu, cache_relu = self.relu_forward(z_fc)
        cache = (cache_fc, cache_relu)
        return z_relu, cache

    def affine_relu_backward(self, dout, cache):
        fc_cache, relu_cache = cache
        da = self.relu_backward(dout, relu_cache)
        dx, dw, db = self.affine_backward(da, fc_cache)
        return dx, dw, db

    def affine_batchnorm_relu_forward(self, x, w, b, gamma, betta, mode, name):
        z_fc, cache_fc = self.affine_forward(x, w, b)
        z_bn, cache_bn = self.batch_norm_forward(z_fc, gamma, betta, mode, name)
        z_relu, cache_relu = self.relu_forward(z_bn)
        cache = (cache_fc, cache_bn, cache_relu)
        return z_relu, cache

    def affine_batchnorm_relu_backward(self, dout, cache):
        fc_cache, cache_bn, relu_cache = cache
        da = self.relu_backward(dout, relu_cache)
        dbn, dgamma, dbetta = self.batch_norm_backward(da, cache_bn)
        dx, dw, db = self.affine_backward(dbn, fc_cache)
        return dx, dw, db, dgamma, dbetta

    def batch_norm_forward(self, x, gamma, betta, mode, name):
        cache = None
        if self.bn_param.get('running_mean' + name) is None:
            self.bn_param['running_mean'+ name] = np.zeros_like(x.shape[0])
            self.bn_param['running_var' + name] = np.zeros_like(x.shape[0])

        running_mean = self.bn_param.get('running_mean'+ name)
        running_var = self.bn_param.get('running_var' + name)
        if mode == "train":
            mu = np.sum(x, axis=0)/x.shape[0]
            sigma = np.sum(np.power(x - mu,2), axis=0)/x.shape[0]
            x_bn = (x - mu)/np.sqrt(sigma + self.epsilon)
            y = gamma * x_bn + betta
            running_mean = 0.9 * running_mean + (1 - 0.9) * mu
            running_var = 0.9 * running_var + (1 - 0.9) * sigma
            cache = (x, x_bn, mu, sigma, gamma, betta)
        else:
            x_bn = (x - running_mean) / np.sqrt(running_var + self.epsilon)
            y = gamma * x_bn + betta
        self.bn_param['running_mean'+name] = running_mean
        self.bn_param['running_var'+name] = running_var

        return y, cache

    def dropout_forward(self, x, p, mode):
        if mode == 'train':
            r = np.random.binomial(1, p, x.shape)/p
            y = r*x
        else:
            y = x
        cache = (x, r)
        return y, cache

    def dropout_backward(self, dout, cache):
        x, r = cache
        return dout*r

    def batch_norm_backward(self, dout, cache):
        x, x_bn, mu, sigma, gamma, betta = cache
        m=x.shape[0]
        dx_bn = dout*gamma
        dgamma = np.sum(dout*x_bn, axis=0)
        dbetta = np.sum(dout, axis=0)
        dsigma = np.sum(dout*(x-mu), axis=0)*-0.5*gamma*np.power(sigma+self.epsilon, -3/2)
        dmu = np.sum(dout * -gamma, axis=0)/np.sqrt(sigma + self.epsilon) + \
                dsigma*-2*np.sum(x-mu, axis=0)/m
        dx = dx_bn/np.sqrt(sigma + self.epsilon) + dsigma*2*(x-mu)/m + dmu/m
        return dx, dgamma, dbetta

    def sgd(self, w, dw, config=None):
        w -= self.lr * dw
        return w

    def sgd_momentum(self, w, dw, name):
        if self.v.get(name) is None:
            self.v[name] = np.zeros_like(w)
        mu = self.config.get("momentum")
        lr = self.config.get("learning_rate")
        self.v[name] = mu * self.v[name] - lr * dw
        w += self.v[name]
        return w

    def nesterov_momentum(self, w, dw, name):
        if self.v.get(name) is None:
            self.v[name] = np.zeros_like(w)
        mu = self.config.get("momentum")
        lr = self.config.get("learning_rate")
        dw_ahead = dw + mu * self.v[name]
        self.v[name] = mu * self.v[name] - lr * dw_ahead
        w += self.v[name]
        return w

    def rmsprop(self, w, dw, name):
        if self.v.get(name) is None:
            self.v[name] = np.zeros_like(w)
        lr = self.config.get("learning_rate")
        dr = self.config.get("decay_rate")
        self.v[name] = dr * self.v[name] + (1 - dr) * dw**2
        w = w - lr * dw / (np.sqrt(self.v[name])+1e-8)
        return w

    def __init__(self, file, lr, hidden_dims, lmbd, C, std=1e-2, batch_size=64, epoch=30, verbose=0, N_train=None, \
                 momentum=0.5, decay_rate=0.99, normalization=1, dropout=1):
        self.config = {"learning_rate" : lr, "momentum" : momentum, "decay_rate" : decay_rate}
        self.normalization = normalization
        self.dropout = dropout
        self.v = {}
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
        self.bn_param = {}
        self.cache = {}
        self.verbose = verbose
        self.lr = lr
        self.lmbd = lmbd
        self.C = C
        self.batch = self.unpickle(file)
        self.batch_size = batch_size
        self.num_epochs = epoch
        self.output_size = self.C
        self.X = np.array(self.batch[b'data'])
        self.y = np.array(self.batch[b'labels'])
        if N_train is None:
            self.N_train = len(self.y)
        else:
            self.N_train = N_train
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X[0:self.N_train, :], self.y[0:self.N_train])
        self.X_train = self.X_train.astype(np.float64)
        self.X_test = self.X_test.astype(np.float64)
        X_scaler = StandardScaler(with_mean=True, with_std=True)
        self.X_train = X_scaler.fit_transform(self.X_train)
        self.X_test = X_scaler.transform(self.X_test)
        #mean_image = np.mean(self.X_train, axis=0)
        #self.X_train = self.X_train.astype(np.float64)
        #self.X_test = self.X_test.astype(np.float64)
        #self.X_train -= mean_image
        #self.X_test -= mean_image
        self.epsilon = 1e-4
        self.input_size = self.X_train.T.shape[0]

        for i in range(self.num_layers):
            if i==0:
                self.params["W" + str(i)] = np.sqrt(2/self.input_size) * np.random.randn(self.input_size, hidden_dims[i])
                # self.params["W" + str(i)] = std * np.random.randn(self.input_size, hidden_dims[i])
                self.params["b" + str(i)] = np.zeros(hidden_dims[i])
                if self.normalization == 1:
                    self.params["gamma" + str(i)] = np.ones(hidden_dims[i])
                    self.params["betta" + str(i)] = np.zeros(hidden_dims[i])

            elif i<self.num_layers-1:
                self.params["W" + str(i)] = np.sqrt(2/hidden_dims[i-1]) * np.random.randn(hidden_dims[i-1], hidden_dims[i])
                # self.params["W" + str(i)] = std * np.random.randn(hidden_dims[i - 1], hidden_dims[i])
                self.params["b" + str(i)] = np.zeros(hidden_dims[i])
                if self.normalization == 1:
                    self.params["gamma" + str(i)] = np.ones(hidden_dims[i])
                    self.params["betta" + str(i)] = np.zeros(hidden_dims[i])
            else:
                self.params["W" + str(i)] = np.sqrt(2/hidden_dims[i-1]) * np.random.randn(hidden_dims[i-1], self.C)
                # self.params["W" + str(i)] = std * np.random.randn(hidden_dims[i - 1], self.C)
                self.params["b" + str(i)] = np.zeros(self.C)

    def fit(self, X, y=None):
        a = {"layer0" : X}
        if y is None:
            mode="test"
        else:
            mode="train"
        for i in range(self.num_layers):
            l, l_prev = 'layer' + str(i + 1), 'layer' + str(i)
            W, b = self.params["W"+str(i)], self.params["b"+str(i)]
            if i < self.num_layers - 1:
                if self.normalization:
                    gamma, betta = self.params["gamma" + str(i)], self.params["betta" + str(i)]
                    a[l], self.cache[l] = self.affine_batchnorm_relu_forward(a[l_prev], W, b, gamma, betta, mode, "layer" + str(i))
                else:
                    a[l], self.cache[l] = self.affine_relu_forward(a[l_prev], W, b)
                if self.dropout:
                    a[l], self.cache[l] = self.dropout_forward(a[l], 0.5, mode)
            else:
                a[l], self.cache[l] = self.affine_forward(a[l_prev], W, b)

        out = a["layer"+str(self.num_layers)]

        if y is None:
            return out

        loss, grad, dout  = 0, {}, {}
        loss, dout = self.softmax_loss(out, y)
        for i in range(self.num_layers):
            w = "W"+str(i)
            loss += 0.5 * self.lmbd * np.sum(self.params[w]**2)

        for i in reversed(range(self.num_layers)):
            l, l_prev = 'layer' + str(i + 1), 'layer' + str(i)
            w, b = "W" + str(i), "b" + str(i)
            gamma, betta = "gamma" + str(i), "betta" + str(i)
            if i==self.num_layers-1:
                dout[l_prev], grad[w], grad[b] = self.affine_backward(dout, self.cache[l])
            elif i<self.num_layers-1:
                if self.dropout == 1:
                    dout[l] = self.dropout_backward(dout[l], self.cache[l])
                if self.normalization:
                    dout[l_prev], grad[w], grad[b], grad[gamma], grad[betta] = \
                        self.affine_batchnorm_relu_backward(dout[l], self.cache[l])
                else:
                    dout[l_prev], grad[w], grad[b] = \
                        self.affine_relu_backward(dout[l], self.cache[l])

        for i in range(self.num_layers):
            w = "W" + str(i)
            grad[w] += self.lmbd * self.params[w]

        return loss, grad

    def train(self):
        loss_story, test_loss_story, train_loss_story, weight_scale_story = [], [], [], []
        num_train = self.X_train.shape[0]
        iterations_per_epoch = np.round(max(num_train / self.batch_size, 1))
        self.num_iter = np.round(self.num_epochs*iterations_per_epoch).astype(np.int)
        num_epoch = 1
        for i in range(self.num_iter):
            rand_range = np.random.randint(0, self.y_train.shape[0], self.batch_size)
            X = self.X_train[rand_range]
            y = self.y_train[rand_range]
            loss, grad = self.fit(X, y)
            loss_story.append(loss)
            for ii in range(self.num_layers):
                w, b = "W" + str(ii), "b" + str(ii)
                self.params[w] = self.rmsprop(self.params[w], grad[w], w)
                self.params[b] = self.rmsprop(self.params[b], grad[b], b)
                if ii<self.num_layers-1 and self.normalization == 1:
                    gamma, betta = "gamma" + str(ii), "betta" + str(ii)
                    self.params[gamma] = self.rmsprop(self.params[gamma], grad[gamma], gamma)
                    self.params[betta] = self.rmsprop(self.params[betta], grad[betta], betta)

            param_scale = np.linalg.norm(self.params["W0"].ravel())
            update_scale = self.config["learning_rate"]*np.linalg.norm(grad["W0"].ravel())/grad["W0"].ravel().shape[0]
            weight_scale_story.append(update_scale)

            if self.verbose and i % 10 == 0:
                print('iteration %d / %d: loss %f' % (i, self.num_iter, loss))
            if i % iterations_per_epoch == 0:
                self.config["learning_rate"] *= 0.7
                train_loss = self.predict(self.X_train, self.y_train, N_train=1000, switch=1)
                test_loss = self.predict(self.X_test, self.y_test, switch=1)
                test_loss_story.append(test_loss)  # np.linalg.norm(self.W2))
                train_loss_story.append(train_loss)  # np.linalg.norm(self.W2))
                print('Epoch %d / %d: train_acc %f; test_acc %f; lr %f' % (num_epoch, self.num_epochs, train_loss, test_loss, self.config["learning_rate"]))
                num_epoch += 1
            #self.progress(n)
        return loss_story, test_loss_story, train_loss_story, weight_scale_story

    def predict(self, X, y, N_train = None, switch=None):
        if N_train is None:
            N_train = X.shape[0]
        y_pred = self.fit(X[:N_train])
        y_pred_10 = y_pred.argmax(1)
        test_acc = accuracy_score(y_pred_10, y[:N_train])
        if (switch is None):
            print('\nTest Accuracy', test_acc)
            return y_pred_10
        else:
            return test_acc

    def test(self):
        num_inputs = 2
        input_shape = (4,5,6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)
        out, _ = self.affine_forward(x, w, b)
        print(out)

        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        _, cache = self.affine_forward(x, w, b)
        dx, dw, db = self.affine_backward(dout, cache)

        print('Testing affine_backward function:')
        print(dx.shape)



ann_clf = FCN(file="./cifar-10-batches-py/data_batch_2", lr=0.4e-3, hidden_dims = [100], lmbd=0.1, C=10, \
              batch_size=100, epoch=30, verbose=0, std=1e-3, N_train=10000, momentum=0.99, decay_rate=0.99, \
              normalization=1, dropout=1)

#ann_clf.test()

loss, test_loss, train_loss, update = ann_clf.train()
print("\n")
print(loss[-1], test_loss[-1], train_loss[-1])

plt.subplot(3, 1, 1)
plt.plot(loss)
plt.ylabel('Loss')
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.tight_layout()
plt.plot(test_loss)
plt.plot(train_loss)
plt.ylabel('Test/test accuracy')
plt.subplot(3, 1, 3)
plt.plot(update)
plt.ylabel('Update')
plt.tight_layout()
plt.show()
