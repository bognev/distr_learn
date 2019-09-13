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
        pass


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    def fit_base(self):
        N_dp = self.X_train.shape[0]
        cost = []
        costs = 0
        # y_est = np.zeros(K)
        for i in range(0, self.num_iter):
            y_est = self.X_train@self.w
            cost = (1 / N_dp) * np.sum(
                np.log(np.exp(self.W.T @ X_T) / (np.ones((C, 1)) * (np.ones((1, C)) @ np.exp(self.W.T @ X_T)))), axis=1)
            cost.append(costs)
            self.dw_1 = 1 + np.exp(self.y_train*(self.X_train@self.w))
            self.dw_2 = self.y_train/self.dw_1
            self.dw_3 = (1/N_dp)*self.X_train.T @ self.dw_2
            self.dw = self.dw_3
            self.w = self.w + self.lr * self.dw
        return cost

    def predict(self):
        N_dp = self.X_test.shape[0]
        y_T = self.y_test
        X_T = self.X_test.T
        y_pred = np.exp(self.W.T @ X_T) / (np.ones((self.C, 1)) * (np.ones((1, self.C)) @ np.exp(self.W.T @ X_T)))
        y_pred_10 = np.argmax(y_pred)
        return y_pred_10


soft_max = SoftMax(file="./cifar-10-batches-py/data_batch_1", lr=0.000001, num_iter=10, lmbd=0.1, C=10)
cost = soft_max.predict()

print(cost)

