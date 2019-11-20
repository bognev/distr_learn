import numpy as np
from random import randrange
from random import sample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.signal import savgol_filter

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

    def sgd(self, w, dw, config=None):
        w -= self.lr * dw
        return w, config

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
        w = w - lr*dw/(np.sqrt(self.v[name])+1e-8)
        return w

    def __init__(self, file, lr, hidden_dims, lmbd, C, std=1e-2, batch_size=64, epoch=100, verbose=0, N_train=None, \
                 momentum=0.5, decay_rate=0.99):
        self.config = {"learning_rate" : lr, "momentum" : momentum, "decay_rate" : decay_rate}
        self.v = {}
        self.use_batchnorm = None
        self.use_dropout = None
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
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
        mean_image = np.mean(self.X_train, axis=0)
        self.X_train = self.X_train.astype(np.float64)
        self.X_test = self.X_test.astype(np.float64)
        self.X_train -= mean_image
        self.X_test -= mean_image
        self.epsilon = 1e-5
        self.input_size = self.X_train.T.shape[0]

        for i in range(self.num_layers):
            if i==0:
                self.params["W" + str(i)] = np.sqrt(2/self.input_size) * np.random.randn(self.input_size, hidden_dims[i])
                self.params["b" + str(i)] = np.zeros(hidden_dims[i])
            elif i<self.num_layers-1:
                self.params["W" + str(i)] = np.sqrt(2/hidden_dims[i-1]) * np.random.randn(hidden_dims[i-1], hidden_dims[i])
                self.params["b" + str(i)] = np.zeros(hidden_dims[i])
            else:
                self.params["W" + str(i)] = np.sqrt(2/hidden_dims[i-1]) * np.random.randn(hidden_dims[i-1], self.C)
                self.params["b" + str(i)] = np.zeros(self.C)

    def fit(self, X, y=None):
        a = {"layer0" : X}
        for i in range(self.num_layers):
            l, l_prev = 'layer' + str(i + 1), 'layer' + str(i)
            W, b = self.params["W"+str(i)], self.params["b"+str(i)]
            if i < self.num_layers - 1:
                a[l], self.cache[l] = self.affine_relu_forward(a[l_prev], W, b)
            else:
                a[l], self.cache[l] = self.affine_forward(a[l_prev], W, b)

        out = a["layer"+str(self.num_layers)]

        if y is None:
            return out

        loss, grad, delta  = 0, {}, {}
        loss, dout = self.softmax_loss(out, y)
        for i in range(self.num_layers):
            w = "W"+str(i)
            loss += 0.5 * self.lmbd * np.sum(self.params[w]**2)

        for i in reversed(range(self.num_layers)):
            l, l_prev = 'layer' + str(i + 1), 'layer' + str(i)
            w, b = "W" + str(i), "b" + str(i)
            if i==self.num_layers-1:
                delta[l_prev], grad[w], grad[b] = self.affine_backward(dout, self.cache[l])
            elif i<self.num_layers-1:
                delta[l_prev], grad[w], grad[b] = self.affine_relu_backward(delta[l], self.cache[l])

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

            param_scale = np.linalg.norm(self.params["W0"].ravel())
            update_scale = np.linalg.norm(grad["W0"].ravel())
            weight_scale_story.append(param_scale/update_scale)

            if self.verbose and i % 10 == 0:
                print('iteration %d / %d: loss %f' % (i, self.num_iter, loss))
            if i % iterations_per_epoch == 0:
                self.config["learning_rate"] *= 1
                train_loss = self.predict(self.X_train, self.y_train, switch=1)
                test_loss = self.predict(self.X_test, self.y_test, switch=1)
                test_loss_story.append(test_loss)  # np.linalg.norm(self.W2))
                train_loss_story.append(train_loss)  # np.linalg.norm(self.W2))
                print('Epoch %d / %d: train_acc %f; test_acc %f' % (num_epoch, self.num_epochs, train_loss, test_loss))
                num_epoch += 1
            #self.progress(n)
        return loss_story, test_loss_story, train_loss_story, weight_scale_story

    def predict(self, X, y, switch=None):
        y_pred = self.fit(X)
        y_pred_10 = y_pred.argmax(1)
        test_acc = accuracy_score(y_pred_10, y)
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



ann_clf = FCN(file="./cifar-10-batches-py/data_batch_1", lr=1e-3, hidden_dims = [100], lmbd=0.05, C=10, \
              batch_size=100, epoch=30, verbose=0, std=1e-3, N_train=10000, momentum=0.5, decay_rate=0.999)

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
