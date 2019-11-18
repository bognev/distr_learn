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

    def __init__(self, file, lr, num_iter, lmbd, C, std=1e-4, batch = 64, epoch = 100, hidden_size = 100):
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
        self.params['W1'] = std * np.random.randn(self.input_size, self.hidden_size)
        self.params['b1'] = std * np.random.randn(self.hidden_size)
        self.params['W2'] = std * np.random.randn(self.hidden_size, self.output_size)
        self.params['b2'] = std * np.random.randn(self.output_size)

        self.W1, self.b1 = self.params['W1'], self.params['b1']
        self.W2, self.b2 = self.params['W2'], self.params['b2']

    def fit(self, X, y=None):
        N = X.T.shape[1]

        z = X@self.W1 + self.b1
        h = np.maximum(z, 0)
        s = h@self.W2 + self.b2

        x = s.T
        shiftx = x - x.max(0)
        shiftx = np.exp(shiftx)
        out = shiftx / np.sum(shiftx, axis=0)
        out = out.T

        if y is None:
            return out

        costs = np.log(out[(np.arange(N), np.array(y))] / np.sum(out, axis=1) + self.epsilon)
        loss = -np.sum(costs) / N
        loss += 0.5 * self.lmbd * (np.sum(self.W1**2) + np.sum(self.W2**2))

        grad = {}
        dout = out
        dout[(np.arange(N), np.array(y))] -= 1
        dh = dout @ self.W2.T
        dz = dh * (z > 0)

        grad['W2'] = h.T @ dout / N
        grad['b2'] = np.sum(dout, axis=0) / N
        grad['W1'] = X.T @ dz / N
        grad['b1'] = np.sum(dz, axis=0) / N

        grad['W2'] += self.lmbd * self.W2
        grad['W1'] += self.lmbd * self.W1

        return loss, grad

    def train(self):
        loss_story = []
        test_loss_story = []
        for n in range(0, self.num_epochs):
            rand_range = np.random.randint(0, len(self.y_train), self.batch)
            X = self.X_train[rand_range]
            y = np.array(self.y_train)[rand_range]
            for i in range(self.num_iter):
                loss, grad = self.fit(X, y)
                self.W2 -= self.lr * grad['W2']
                self.W1 -= self.lr * grad['W1']
                self.b2 -= self.lr * grad['b2']
                self.b1 -= self.lr * grad['b1']
            self.lr *= 0.999
            test_loss = self.predict(switch=1)
            loss_story.append(loss)
            test_loss_story.append(test_loss)#np.linalg.norm(self.W2))
            self.progress(n)
        return loss_story, test_loss_story

    def predict(self, switch=None):
        y = self.y_test
        X = self.X_test
        y_pred = self.fit(X)
        y_pred_10 = y_pred.argmax(1)
        test_acc = accuracy_score(y_pred_10, y)
        if(switch is None):
            print('\nTest Accuracy', test_acc)
            return y_pred_10
        else:
            return test_acc


ann_clf = ANN(file="./cifar-10-batches-py/data_batch_1", lr=1e-3, num_iter=10, \
              lmbd=0.001, C=10, batch=128, epoch=500, hidden_size=100)
# np.random.seed(10)
loss, test_loss = ann_clf.train()
norm = 0
y_pred = ann_clf.predict()

plt.subplot(2, 1, 1)
plt.plot(loss)

plt.tight_layout()
plt.ylabel('cost')
plt.subplot(2, 1, 2)
plt.plot(test_loss)
# plt.tight_layout()
# plt.ylabel('norm')
# plt.ylabel('norm_z')
# plt.xlabel('epochs')
plt.tight_layout()
plt.show()



