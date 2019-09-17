import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

class SoftMax:
    def __init__(self, file, lr, num_iter, lmbd, C):
        self.lr = lr
        self.num_iter = num_iter
        self.lmbd = lmbd
        self.C = C
        self.batch = self.unpickle(file)
        self.X = self.batch[b'data']
        self.y = self.batch[b'labels']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        N_dp = self.X_test.shape[0]
        self.X_test = np.column_stack((self.X_test, np.ones(N_dp)))
        N_dp = self.X_train.shape[0]
        self.X_train = np.column_stack((self.X_train, np.ones(N_dp)))
        self.n_features = self.X_test.shape[1]
        self.W = np.zeros((self.n_features, self.C))
        self.dW = np.zeros((self.n_features, self.C))


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    def stablesoftmax(x):
        """Compute the softmax of vector x in a numerically stable way."""
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def fit_base(self):
        N_dp = self.X_train.shape[0]
        y_T = self.y_train
        X_T = self.X_train.T
        cost = np.zeros(int(self.num_iter/10))
        costs = np.zeros(N_dp)
        dW_1 = np.zeros((self.C,1))
        dW_2 = np.zeros((self.n_features, self.C))
        dW_3 = np.zeros((self.n_features, self.C))
        epsilon = 1e-5
        # y_est = np.zeros(K)
        for i in range(0, self.num_iter):
            if i%10 == 0:
                for n in range(N_dp):
                    x = self.W.T @ X_T[:, n]
                    shiftx = x - np.max(x)
                    shiftx = np.exp(shiftx)
                    costs[n] = np.log(shiftx[self.y_train[n]]/np.sum(shiftx) + epsilon)
                cost[int(i/10)] = -(1 / N_dp)*np.sum(costs)
            dW_3 = np.zeros((self.n_features, self.C))
            for n in range(N_dp):
                x = self.W.T @ X_T[:, n]
                shiftx = x - np.max(x)
                shiftx = np.exp(shiftx)
                dW_1 = shiftx/np.sum(shiftx)
                dW_1[self.y_train[n]] = dW_1[self.y_train[n]] - 1
                dW_2 = dW_1.reshape(10,1)@X_T[:,n].reshape(1,3073) + 0.001*self.W.T
                dW_3 = dW_2.T + dW_3
            self.dW = (1/N_dp)*dW_3
            self.W = self.W - self.lr * self.dW
        return cost

    def fit_vec_base(self):
        N_dp = self.X_train.shape[0]
        y_T = self.y_train
        X_T = self.X_train.T
        cost = np.zeros(int(self.num_iter))
        norm = np.zeros(int(self.num_iter))
        costs = np.zeros(N_dp)
        dW_1 = np.zeros((self.C,1))
        dW_2 = np.zeros((self.n_features, self.C))
        dW_3 = np.zeros((self.n_features, self.C))
        epsilon = 1e-5
        # y_est = np.zeros(K)
        for i in range(0, self.num_iter):
            # if i%10 == 0:
            x = self.W.T @ X_T
            shiftx = x - x.max(0)
            shiftx = np.exp(shiftx)
            dW_1 = shiftx/np.sum(shiftx, axis=0)
            dW_1[(np.array(self.y_train),np.arange(0,N_dp))] = dW_1[(np.array(self.y_train),np.arange(0,N_dp))] - 1
            dW_2 = dW_1@self.X_train + 0.001*self.W.T
            self.dW = (1/N_dp)*dW_2
            self.W = self.W - self.lr * self.dW.T
            costs = np.log(shiftx[(np.array(self.y_train), np.arange(0, N_dp))] / np.sum(shiftx, axis=0) + epsilon)
            cost[i] = -(1 / N_dp) * np.sum(costs)
            norm[i] = np.linalg.norm(self.W)
        return cost, norm

    def predict(self):
        N_dp = self.X_test.shape[0]
        y_T = self.y_test
        X_T = self.X_test.T
        y_pred = np.exp(self.W.T @ X_T) / (np.ones((self.C, 1)) * (np.ones((1, self.C)) @ np.exp(self.W.T @ X_T)))
        y_pred_10 = y_pred.argmax(0)
        print('Test Accuracy', accuracy_score(y_pred_10, self.y_test))
        return y_pred_10


soft_max = SoftMax(file="./cifar-10-batches-py/data_batch_1", lr=0.0000001, num_iter=5000, lmbd=0.1, C=10)
cost, norm = soft_max.fit_vec_base()
y_pred = soft_max.predict()

# print(cost)

plt.subplot(2, 1, 1)
plt.plot(cost)
plt.subplot(2, 1, 2)
plt.plot(norm)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title('Cost reduction over time')
plt.show()



