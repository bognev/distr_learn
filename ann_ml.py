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


class ANN:
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

    def __init__(self, file, lr, num_iter, lmbd, C, std=1e-2, batch=64, epoch=100, hidden_size=100, verbose=0):
        self.verbose = verbose
        self.lr = lr
        self.num_iter = num_iter
        self.lmbd = lmbd
        self.C = C
        self.batch = self.unpickle(file)
        self.X = self.batch[b'data']
        self.y = self.batch[b'labels']
        self.batch = batch
        self.num_epochs = epoch
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        mean_image = np.mean(self.X_train, axis=0)
        self.X_train = self.X_train.astype(np.float64)
        self.X_test = self.X_test.astype(np.float64)
        self.X_train -= mean_image
        self.X_test -= mean_image
        self.epsilon = 1e-5

        self.input_size = self.X_train.T.shape[0]
        self.hidden_size = hidden_size
        self.output_size = self.C

        self.params = {}
        self.W1 = std * np.random.randn(self.input_size, self.hidden_size)#np.sqrt(2.0/self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = std * np.random.randn(self.hidden_size, self.output_size)#np.sqrt(2.0/self.hidden_size)
        self.b2 = np.zeros(self.output_size)

        #self.W1, self.b1 = self.params['W1'], self.params['b1']
        #self.W2, self.b2 = self.params['W2'], self.params['b2']

        self.cache = {}

    def affine_forward(self, x, w, b):
        N = x.shape[0]
        z = None
        X = np.reshape(x, (N, -1))
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

    def fit(self, X, y=None):
        hidden, self.cache["hidden"] = self.affine_relu_forward(X, self.W1, self.b1)
        out, self.cache["out"] = self.affine_forward(hidden, self.W2, self.b2)

        if y is None:
            return out

        loss, grad = 0, {}
        loss, delta3 = self.softmax_loss(out, y)
        loss += 0.5 * self.lmbd * (np.sum(self.W1**2) + np.sum(self.W2**2))

        delta2, grad["W2"], grad["b2"] = self.affine_backward(delta3, self.cache["out"])
        delta1, grad["W1"], grad["b1"] = self.affine_relu_backward(delta2, self.cache["hidden"])

        grad['W2'] += self.lmbd * self.W2
        grad['W1'] += self.lmbd * self.W1

        return loss, grad

    def train(self):
        loss_story, test_loss_story, train_loss_story = [], [], []
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch, 1)
        num_epoch = 1
        for i in range(self.num_iter):
            rand_range = np.random.randint(0, len(self.y_train), self.batch)
            X = self.X_train[rand_range]
            y = np.array(self.y_train)[rand_range]
            loss, grad = self.fit(X, y)
            loss_story.append(loss)
            self.W2 -= self.lr * grad['W2']
            self.W1 -= self.lr * grad['W1']
            self.b2 -= self.lr * grad['b2']
            self.b1 -= self.lr * grad['b1']
            #self.params['W1'], self.params['b1'] = self.W1, self.b1
            #self.params['W2'], self.params['b2'] = self.W2, self.b2
            if self.verbose and i % 25 == 0:
                print('iteration %d / %d: loss %f' % (i, self.num_iter, loss))
            if i % iterations_per_epoch == 0:
                self.lr *= 0.95
                train_loss = self.predict(self.X_train, self.y_train, switch=1)
                test_loss = self.predict(self.X_test, self.y_test, switch=1)
                test_loss_story.append(test_loss)  # np.linalg.norm(self.W2))
                train_loss_story.append(train_loss)  # np.linalg.norm(self.W2))
                print('Epoch %d / %d: train_acc %f; test_acc %f' % (num_epoch, self.num_epochs, train_loss, test_loss))
                num_epoch += 1
            #self.progress(n)
        return loss_story, test_loss_story, train_loss_story

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



ann_clf = ANN(file="./cifar-10-batches-py/data_batch_1", lr=1e-3, num_iter=7500, \
              lmbd=5, C=10, batch=100, epoch=100, hidden_size=100, verbose=0)

#ann_clf.test()

loss, test_loss, train_loss = ann_clf.train()
print("\n")
print(loss[-1], test_loss[-1], train_loss[-1])

plt.subplot(3, 1, 1)
plt.plot(loss)
plt.ylabel('Loss')
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.tight_layout()
plt.plot(test_loss)
plt.ylabel('Test accuracy')
plt.subplot(3, 1, 3)
plt.plot(train_loss)
plt.ylabel('Train accuracy')
plt.tight_layout()
plt.show()
