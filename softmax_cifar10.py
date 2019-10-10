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
        self.W = np.zeros((self.n_features+2, self.C))
        self.dW = np.zeros((self.n_features+2, self.C))


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    def network_topology(self, K):
        # self.G1 = np.array([[1,0,0,1,1],
        #                     [0,1,1,0,0],
        #                     [0,0,1,1,0],
        #                     [0,0,0,1,0],
        #                     [0,0,0,0,1]])
        # self.G1 = np.array([[1,0,0,0,0],
        #                     [0,1,0,0,0],
        #                     [0,0,1,0,0],
        #                     [0,0,0,1,0],
        #                     [0,0,0,0,1]])
        # self.G1 = np.array([[1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        #                     [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #                     [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        #                     [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        #                     [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        #                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        #                     [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        #                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.G1 = np.array([[1, 1, 1, 0, 1],
                            [0, 1, 0, 1, 1],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]])
        self.G1 = self.G1 + self.G1.T - np.eye(K)
        self.G = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                if self.G1[i,j] == 1:
                    if i==j:
                        pass
                    else:
                        self.G[i, j] = 1/(max(sum(self.G1[i,:])-1, sum(self.G1[j,:])-1)+1)
                if j == K-1:
                    self.G[i,i] = 1 - sum(self.G[i,:])
        print(self.G.round(decimals=3))
        pass

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
        cost = np.zeros(int(self.num_iter))
        norm = np.zeros(int(self.num_iter))
        costs = np.zeros(N_dp)
        dW_1 = np.zeros((self.C,1))
        dW_2 = np.zeros((self.n_features, self.C))
        dW_3 = np.zeros((self.n_features, self.C))
        epsilon = 1e-5
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        y_T = self.y_train
        X_T = self.X_train.T
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

    def fit_vec_batch(self, N_batch=250, N_epoch=250):
        N_dp = N_batch#self.X_train.shape[0]
        self.num_epochs = N_epoch
        cost = np.zeros(int(self.num_epochs))
        norm = np.zeros(int(self.num_epochs))
        costs = np.zeros(N_dp)
        dW_1 = np.zeros((self.C,1))
        dW_2 = np.zeros((self.n_features, self.C))
        dW_3 = np.zeros((self.n_features, self.C))
        epsilon = 1e-5
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        # y_est = np.zeros(K)
        self.y_train = np.array(self.y_train)
        for n in range(0, self.num_epochs):
            rand_range = np.random.randint(0, len(self.y_train), N_dp).tolist()
            X_T = self.X_train[rand_range].T
            for i in range(0, self.num_iter):
                # if i%10 == 0:
                x = self.W.T @ X_T
                shiftx = x - x.max(0)
                shiftx = np.exp(shiftx)
                dW_1 = shiftx/np.sum(shiftx, axis=0)
                dW_1[(self.y_train[rand_range],np.arange(0,N_dp))] -=  1
                dW_2 = dW_1@self.X_train[rand_range] + self.lmbd*self.W.T
                self.dW = (1/N_dp)*dW_2
                self.W = self.W - self.lr/np.sqrt(n+1) * self.dW.T
            costs = np.log(shiftx[(self.y_train[rand_range], np.arange(0, N_dp))] / np.sum(shiftx, axis=0) + epsilon)
            cost[n] = -(1 / N_dp) * np.sum(costs)
            norm[n] = np.linalg.norm(self.W)
        return cost, norm

    def fit_vec_batch_alg1(self, K, N_batch=250, N_epoch=250):
        self.network_topology(K)
        N_dp = N_batch#self.X_train.shape[0]
        self.num_epochs = N_epoch
        self.n_features +=2
        cost = np.zeros(int(self.num_epochs))
        norm = np.zeros(int(self.num_epochs))
        norm_z = np.zeros(int(self.num_epochs))
        costs = np.zeros(N_dp)
        dW_1 = np.zeros((self.C,1))
        dW_2 = np.zeros((self.n_features, self.C))
        dW_3 = np.zeros((self.n_features, self.C))
        epsilon = 1e-8
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        self.X_train = np.column_stack((self.X_train, np.zeros(self.X_train.shape[0])))
        self.w_distr = np.zeros((K, self.C, int(round(self.n_features / K))))
        z_ni_t = np.zeros((K, self.C, N_batch))
        h_nik = np.zeros((K, N_batch, int(round(self.n_features / K))))
        self.y_train = np.array(self.y_train)
        self.W = np.zeros((self.C, self.n_features))
        for n in range(0, self.num_epochs):
            ni = np.random.randint(0, len(self.y_train), N_dp).tolist()
            z_ni_t = np.zeros((K, self.C, N_batch))
            # consensus update
            for i in range(0, self.num_iter):
                if n == 1:
                    for kk in range(K):
                        z_ni_t[kk] = self.w_distr[kk] @ \
                                     self.X_train[np.array(ni)[:, None], \
                                                    np.arange(kk * int(round(self.n_features / K)), \
                                                                (kk + 1) * int(round(self.n_features / K)))].T
                elif(n > 1):
                    for kk in range(K):
                        for ll in range(K):
                            z_ni_t[kk] += self.G[kk, ll] * self.w_distr[ll] @ \
                                              self.X_train[np.array(ni)[:, None], \
                                                           np.arange(ll * int(round(self.n_features / K)), \
                                                                     (ll + 1) * int(round(self.n_features / K)))].T
            z_ni_t = K * z_ni_t
            for kk in range(K):
                h_nik[kk] = self.X_train[np.array(ni)[:, None], \
                                                 np.arange(kk * int(round(self.n_features / K)), \
                                                           (kk + 1) * int(round(self.n_features / K)))]
                x = z_ni_t[kk]
                shiftx = x - x.max(0)
                shiftx = np.exp(shiftx)
                dW_1 = shiftx/np.sum(shiftx, axis=0)
                dW_1[(self.y_train[ni],np.arange(0,N_dp))] -= 1
                dW_2 = dW_1 @ h_nik[kk] + self.lmbd * self.w_distr[kk]
                self.dW = (1/N_dp) * dW_2
                self.w_distr[kk] = self.w_distr[kk] - self.lr/np.sqrt(n+1) * self.dW
            norm_z[n] = np.linalg.norm(z_ni_t)
            for kk in range(K):
                self.W[:, np.arange(kk * int(round(self.n_features / K)), (kk + 1) * int(round(self.n_features / K)))] = \
                    self.w_distr[kk]
            X_T = self.X_train[ni].T
            x = self.W @ X_T
            shiftx = x - x.max(0)
            shiftx = np.exp(shiftx)
            costs = np.log((shiftx[(self.y_train[ni],np.arange(0, N_dp))] / np.sum(shiftx, axis=0)) + epsilon)
            cost[n] = -(1 / N_dp) * np.sum(costs)
            norm[n] = np.linalg.norm(self.W)
        return cost, norm, norm_z

    def predict(self):
        N_dp = self.X_test.shape[0]
        y_T = self.y_test
        X_T = self.X_test.T
        y_pred = np.exp(self.W[np.arange(0, self.n_features),:].T @ X_T) / (np.ones((self.C, 1)) * (np.ones((1, self.C)) @ np.exp(self.W[np.arange(0, self.n_features),:].T @ X_T)))
        y_pred_10 = y_pred.argmax(0)
        print('Test Accuracy', accuracy_score(y_pred_10, self.y_test))
        return y_pred_10

    def predict_alg1(self):
        N_dp = self.X_test.shape[0]
        y_T = self.y_test
        X_T = self.X_test.T
        y_pred = np.exp(self.W[:, np.arange(0, self.n_features - 2)] @ X_T) / (
                    np.ones((self.C, 1)) * (np.ones((1, self.C)) @ np.exp(self.W[:, np.arange(0, self.n_features - 2)] @ X_T)))
        y_pred_10 = y_pred.argmax(0)
        print('Test Accuracy', accuracy_score(y_pred_10, self.y_test))
        return y_pred_10

soft_max = SoftMax(file="./cifar-10-batches-py/data_batch_1", lr=1e-7, num_iter=1, lmbd=1e-03, C=10)
soft_max_alg1 = SoftMax(file="./cifar-10-batches-py/data_batch_1", lr=0.2e-7, num_iter=1, lmbd=1e-03, C=10)
np.random.seed(10)
cost_alg1, norm_alg1, norm_z_alg1 = soft_max_alg1.fit_vec_batch_alg1(K=5, N_epoch=10000, N_batch=50)
np.random.seed(10)
cost, norm = soft_max.fit_vec_batch(N_epoch=3000, N_batch=20)#_alg1(K=15, N_epoch=1600, N_batch=10)

y_pred = soft_max.predict()
y_pred_alg1 = soft_max_alg1.predict_alg1()

# print(cost)

plt.subplot(4, 1, 1)
plt.plot(cost)
plt.plot(savgol_filter(cost, 151, 3))
plt.tight_layout()
plt.ylabel('cost')
plt.subplot(4, 1, 2)
plt.plot(norm)
plt.tight_layout()
plt.ylabel('norm')
plt.subplot(4, 1, 3)
plt.plot(cost_alg1)
plt.plot(savgol_filter(cost_alg1, 151, 3))
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.plot(norm_alg1)
plt.tight_layout()
plt.ylabel('norm_z')
plt.xlabel('epochs')
plt.tight_layout()
plt.show()



