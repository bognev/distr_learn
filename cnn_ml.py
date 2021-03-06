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
from fast_layers import *
from layer_utils import *
from layers import relu_forward, relu_backward
style.use("ggplot")
import sys
from PIL import Image
from optim_self import adam


class CNN:
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    def dump_pickle(self, file, dic):
        with open(file, 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


    def __init__(self, file, lr, input_dim = (3,32,32), hidden_dim=100, num_filters=32, filter_size=7, lmbd=0.01, C=10, std=1e-2, batch_size=64, epoch=30, verbose=0, N_train=None, \
                 momentum=0.5, decay_rate=0.99):
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.config = {"learning_rate" : lr, "momentum" : momentum, "decay_rate" : decay_rate}
        self.config.setdefault("beta1", 0.9)
        self.config.setdefault("beta2", 0.999)
        self.v = {}
        self.num_layers = 3
        self.params = {}
        self.bn_param = {}
        self.cache = {}
        self.verbose = verbose
        self.lr = lr
        self.lmbd = lmbd
        self.C = C
        # self.batch1 = self.unpickle(file + "1")
        # self.batch2 = self.unpickle(file + "2")
        # self.batch3 = self.unpickle(file + "3")
        # self.batch4 = self.unpickle(file + "4")
        # self.batch5 = self.unpickle(file + "5")
        # self.batch1 = self.unpickle("file + 1")
        # self.batch2 = self.unpickle("file + 2")
        # self.batch3 = self.unpickle("file + 3")
        # self.batch4 = self.unpickle("file + 4")
        # self.batch5 = self.unpickle("file + 5")
        # self.labels1 = self.unpickle("labels + 1")
        # self.labels2 = self.unpickle("labels + 2")
        # self.labels3 = self.unpickle("labels + 3")
        # self.labels4 = self.unpickle("labels + 4")
        # self.labels5 = self.unpickle("labels + 5")
        # self.dump_pickle('file + 1', self.batch1[b'data'].reshape((10000, 3, 32, 32)))
        # self.dump_pickle('file + 2', self.batch2[b'data'].reshape((10000, 3, 32, 32)))
        # self.dump_pickle('file + 3', self.batch3[b'data'].reshape((10000, 3, 32, 32)))
        # self.dump_pickle('file + 4', self.batch4[b'data'].reshape((10000, 3, 32, 32)))
        # self.dump_pickle('file + 5', self.batch5[b'data'].reshape((10000, 3, 32, 32)))
        # self.dump_pickle('labels + 1', self.batch1[b'labels'])
        # self.dump_pickle('labels + 2', self.batch2[b'labels'])
        # self.dump_pickle('labels + 3', self.batch3[b'labels'])
        # self.dump_pickle('labels + 4', self.batch4[b'labels'])
        # self.dump_pickle('labels + 5', self.batch5[b'labels'])
        self.batch_size = batch_size
        self.num_epochs = epoch
        self.output_size = self.C
        # self.X = np.concatenate((self.batch1, self.batch2, self.batch3, self.batch4, self.batch5))
        # self.y = np.concatenate((self.labels1, self.labels2, self.labels3, self.labels4, self.labels5))
        # self.dump_pickle('file_batch',  self.X)
        # self.dump_pickle('file_labels', self.y)
        self.X = self.unpickle('file_batch')
        self.y = self.unpickle('file_labels')
        if N_train is None:
            self.N_train = len(self.y)
        else:
            self.N_train = N_train
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X[0:self.N_train, :], self.y[0:self.N_train])
        self.X_train = self.X_train.astype("uint8")
        self.X_test = self.X_test.astype("uint8")
        # X_scaler = StandardScaler(with_mean=True, with_std=True)
        # self.X_train = X_scaler.fit_transform(self.X_train)
        # self.X_test = X_scaler.transform(self.X_test)
        # self.X_train = self.X_train.reshape((self.X_train.shape[0], 3, 32, 32))
        img = Image.fromarray(self.X_train[0].transpose(1,2,0).astype("uint8"), 'RGB')
        img.save('my.png')
        img.show()
        # self.X_train = self.X_train.reshape((self.X_train.shape[0], 3, 32, 32))
        # self.X_test = self.X_test.reshape((self.X_test.shape[0], 3, 32, 32))
        #mean_image = np.mean(self.X_train, axis=0)
        #self.X_train = self.X_train.astype(np.float64)
        #self.X_test = self.X_test.astype(np.float64)
        #self.X_train -= mean_image
        #self.X_test -= mean_image
        self.config["epsilon"] = 1e-8
        self.input_size = self.X_train.shape[0]
        self.conv_param = {'stride': 1, 'pad': int((self.filter_size - 1) / 2)}
        self.pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}

        self.layer0_out_size = int((32 + 2 * self.conv_param['pad'] - self.filter_size) / self.conv_param['stride'] + 1)
        self.layer1_out_size = int((self.layer0_out_size - self.pool_param['pool_height']) / self.pool_param['stride'] + 1)

        # self.params["W0"] = np.sqrt(2 / (self.num_filters*self.filter_size**2)) * np.random.randn(self.num_filters, self.input_dim[0], self.filter_size, self.filter_size)
        self.params["W0"] = std * np.random.randn(self.num_filters, self.input_dim[0], self.filter_size, self.filter_size)
        self.params["b0"] = np.zeros(self.num_filters)
        # self.params["W1"] = np.sqrt(2 / (self.num_filters*self.layer1_out_size**2)) * np.random.randn(self.num_filters*self.layer1_out_size**2, self.hidden_dim)
        self.params["W1"] = std * np.random.randn(self.num_filters * self.layer1_out_size ** 2, self.hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        # self.params["W2"] = np.sqrt(2 / (hidden_dim)) * np.random.randn(self.hidden_dim, self.C)
        self.params["W2"] = std * np.random.randn(self.hidden_dim, self.C)
        self.params["b2"] = np.zeros(self.C)

    def fit(self, X, y=None):

        W1, b1 = self.params['W0'], self.params['b0']
        W2, b2 = self.params['W1'], self.params['b1']
        W3, b3 = self.params['W2'], self.params['b2']
        # print(X.shape)
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)

        out = scores

        if y is None:
            return out

        loss,dx1 = softmax_loss(scores, y)
        loss += 0.5*self.lmbd*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
        grads = {}

        dout1, grads['W2'], grads['b2'] = affine_backward(dx1, cache3)
        grads['W2'] += 2 * self.lmbd * self.params['W2']
        dout2, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache2)
        grads['W1'] += 2 * self.lmbd * self.params['W1']
        _, grads['W0'], grads['b0'] = conv_relu_pool_backward(dout2, cache1)
        grads['W0'] += 2 * self.lmbd * self.params['W0']


        return loss, grads

    def train(self):
        loss_story, test_loss_story, train_loss_story, weight_scale_story = [], [], [], []
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        self.num_iter = self.num_epochs*iterations_per_epoch
        num_epoch = 1
        for i in range(self.num_iter):
            rand_range = np.random.randint(0, self.y_train.shape[0], self.batch_size)
            X = self.X_train[rand_range]
            y = self.y_train[rand_range]
            # print(X.shape)
            loss, grad = self.fit(X, y)
            loss_story.append(loss)
            for ii in range(self.num_layers):
                w, b = "W" + str(ii), "b" + str(ii)
                self.params[w], self.config = adam(self.params[w], grad[w], w, self.config)
                self.params[b], self.config = adam(self.params[b], grad[b], b, self.config)

            param_scale = np.linalg.norm(self.params["W0"].ravel())
            update_scale = self.config["learning_rate"]*np.linalg.norm(grad["W0"].ravel())/grad["W0"].ravel().shape[0]
            weight_scale_story.append(update_scale)

            if self.verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, self.num_iter, loss))
            if i % iterations_per_epoch == 0:
                self.config["learning_rate"] *= 0.95
                train_loss, test_loss = 0,0
                train_loss = self.predict(self.X_train, self.y_train, N_train=100, switch=1)
                test_loss = self.predict(self.X_test, self.y_test, N_train=1000, switch=1)
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





def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


cnn_clf = CNN(file="./cifar-10-batches-py/data_batch_", input_dim = (3,32,32), hidden_dim=500, num_filters=32, filter_size=7, lr=1e-3, lmbd=0.001, C=10, \
              batch_size=50, epoch=30, verbose=1, std=1e-3, N_train=50000, momentum=0.99, decay_rate=0.99)

#ann_clf.test()
# x_shape = (4, 3, 7, 7)
# w_shape = (2, 3, 3, 3)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=2)

# conv_param = {'stride': 2, 'padding': 1}
# pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
# out, cache = cnn_clf.conv_forward_naive(x, w, b, conv_param)
# correct_out = np.array([[[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216 ]],
#                           [[ 0.21027089,  0.21661097],
#                            [ 0.22847626,  0.23004637]],
#                           [[ 0.50813986,  0.54309974],
#                            [ 0.64082444,  0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[ 0.69108355,  0.66880383],
#                            [ 0.59480972,  0.56776003]],
#                           [[ 2.36270298,  2.36904306],
#                            [ 2.38090835,  2.38247847]]]]])
loss, test_loss, train_loss, update = cnn_clf.train()
# pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
# out, cache = cnn_clf.max_pool_forward_naive(out, pool_param)
# cnn_clf.max_pool_backward(out, cache)
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
