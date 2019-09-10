import mnist
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from random import sample
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

#load datasets using https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#plotting one image
# img = Image.fromarray(train_images[0,:,:]* -1 + 256)
# img.save('my.png')
# img.show()

#sklearn
# train_images_r = train_images.reshape(60000,28*28)
# test_images_r = test_images.reshape(10000,28*28)
# logistic = LogisticRegression(n_jobs=-1)
# clf = logistic.fit(X=train_images_r[0:1000], y=train_labels[0:1000]).score(test_images_r[0:100], test_labels[0:100])
# print(logistic.predict(test_images_r[1001:10010]))
# print(test_labels[1001:10010])
# print(clf)
#end sklearn

#select only 0 and 1 labels
train_labels_01 = train_labels[(train_labels == 0) | (train_labels == 1)]
train_images_01 = train_images[(train_labels == 0) | (train_labels == 1)]
train_images_01 = train_images_01.reshape(len(train_labels_01), 28*28)
test_labels_01 = test_labels[(test_labels == 0) | (test_labels == 1)]
test_images_01 = test_images[(test_labels == 0) | (test_labels == 1)]
test_images_01 = test_images_01.reshape(len(test_labels_01), 28*28)

train_labels_01 = np.where(train_labels_01==0, -1, train_labels_01)
test_labels_01 = np.where(test_labels_01==0, -1, test_labels_01)
print(train_labels_01.shape )
print(train_images_01.shape )
print(test_labels_01.shape )
print(test_images_01.shape )

class Log_Reg:
    def __init__(self, lr, num_iter, lmbd):
        self.lr = lr
        self.num_iter = num_iter
        self.lmbd = lmbd

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, X_train, y_train):
        epsilon = 1e-5
        m = X_train.shape[1]
        self.w = np.zeros((1, m))
        self.b = 0
        self.costs = np.zeros(self.num_iter)
        y_T = y_train.T
        X_T = X_train.T
        for i in range(self.num_iter):
            y_estimation = self.sigmoid(np.dot(self.w, X_T) + self.b)
            cost = (-1 / m) * (np.sum(y_T * np.log(y_estimation + epsilon) + (1 - y_T) * (np.log(1 - y_estimation + epsilon))))
            self.dw = (1 / m) * (np.dot(X_T, (y_estimation - y_T).T))
            self.db = (1 / m) * (np.sum(y_estimation - y_T))
            self.w = self.w - self.lr * (self.dw.T + self.lmbd*self.w)
            self.b = self.b - self.lr * self.db
            self.costs[i] = cost
        return self.costs

    def predict(self, X_test, y_test):
        n_features = X_test.shape[1]
        N_dp = X_test.shape[0]
        X_test = np.column_stack((X_test, np.ones(N_dp)))
        y_T = y_test.T
        X_T = X_test.T
        y_pred = self.sigmoid(np.dot(self.w, X_T))
        y_pred_bin = np.zeros(y_pred.shape[0])
        for i in range(y_pred.shape[0]):
            if (y_pred[i] > 0.5):
                y_pred_bin[i] = 1
            else:
                y_pred_bin[i] = -1
        return y_pred_bin

    def performance(self, y_est, y_gt):
        TP, TN, FP, FN, P, N = 0, 0, 0, 0, 0, 0
        for idx, test_sample in enumerate(y_est):
            if y_est[idx] == 1 and test_sample == 1:
                TP = TP + 1
                P = P + 1
            elif y_est[idx] == 1 and test_sample == 0:
                FP = FP + 1
                N = N + 1
            elif y_est[idx] == 0 and test_sample == 0:
                TN = TN + 1
                N = N + 1
            elif y_est[idx] == 0 and test_sample == 1:
                FN = FN + 1
                P = P + 1

        accuracy = (TP + TN)/(P + N)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f_measure = 2 * (precision*recall/(precision + recall))
        return accuracy, precision, recall, f_measure

    def network_topology(self, K):
        self.G1 = np.array([[1,1,0,1,0],
                            [0,1,1,0,1],
                            [0,0,1,0,0],
                            [0,0,0,1,1],
                            [0,0,0,0,1]])
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

    def fit_alg1(self, K, X_train, y_train):
        self.network_topology(K)
        n_features = X_train.shape[1]
        N_dp = X_train.shape[0]
        X_train = np.column_stack((X_train, np.ones(N_dp)))
        n_features = n_features + 1
        self.w_distr = np.zeros((K, int((n_features)/K)))
        cost = []
        costs = 0
        y_est = 0
        z_ni_t = np.zeros(K)
        for i in range(0, self.num_iter):
            ni = sample(range(1,N_dp), 1)[0]
            for k in range(K):
                #consensus update
                for l in range(K):
                    if self.G1[k, l] != 0:
                        z_ni_t[k] = z_ni_t[k] + self.G[k,l]*(X_train[ni, range(l*int(n_features / K),(l + 1) * int(n_features / K))]@self.w_distr[l,:])
                #select features distributed across agents
                h_nik = np.zeros((K, int(n_features / K)))
                h_nik[k, :] = X_train[ni,range(k*int(n_features/K),(k + 1)*int(n_features / K))]
                #compute weights distributed across agents
                self.dw_1 = 1 + np.exp(y_train[ni] * z_ni_t[k])
                self.dw_2 = y_train[ni] / self.dw_1
                self.dw_3 = h_nik[k,:] * self.dw_2
                self.dw = self.dw_3
                self.w_distr[k,:] = self.w_distr[k,:] + self.lr*self.dw# - \
                                    #self.lr*self.lmbd*(self.w_distr[k,:])
            y_est = X_train @ self.w_distr.reshape(n_features)
            costs = (1 / N_dp) * np.sum(np.log(1 + np.exp(-y_train * y_est)))
            cost.append(costs)
            # print(costs)
        return cost

    def fit_base(self, K, X_train, y_train):
        n_features = X_train.shape[1]
        N_dp = X_train.shape[0]
        X_train = np.column_stack((X_train, np.ones(N_dp)))
        self.w = np.zeros(n_features+1)
        cost = []
        costs = 0
        y_est = np.zeros(K)
        for i in range(0, self.num_iter):
            y_est = X_train@self.w
            costs = (1/N_dp)*np.sum(np.log(1+np.exp(-y_train*y_est)))
            cost.append(costs)
            self.dw_1 = 1 + np.exp(y_train*(X_train@self.w))
            self.dw_2 = y_train/self.dw_1
            self.dw_3 = (1/N_dp)*X_train.T @ self.dw_2
            self.dw = self.dw_3
            self.w = self.w + self.lr * self.dw
        return cost


    def predict_alg1(self, X_test, y_test, K):
        y_est = np.zeros(K)
        N_dp = X_test.shape[0]
        X_test = np.column_stack((X_test, np.ones(N_dp)))
        n_features = X_test.shape[1]
        y_pred = np.zeros(N_dp)
        for n in range(N_dp):
            for kk in range(K):
                y_est[kk] = X_test[n, range(kk * int(n_features / K), (kk + 1) * int(n_features / K))]@self.w_distr[kk, :]
            y_pred[n] = self.sigmoid(sum(y_est))
        y_pred_bin = np.zeros(y_pred.shape[0])
        for i in range(y_pred.shape[0]):
            if (y_pred[i] > 0.5):
                y_pred_bin[i] = 1
            else:
                y_pred_bin[i] = -1
        return y_pred_bin



log_reg_m = Log_Reg(lr=0.000001, num_iter=200, lmbd=0.1)
# costs = log_reg_m.fit(X_train=train_images_01, y_train=train_labels_01)
costs = log_reg_m.fit_base(K=8, X_train=train_images_01, y_train=train_labels_01)
costs_distributed = log_reg_m.fit_alg1(K=5, X_train=train_images_01, y_train=train_labels_01)
# y_train = log_reg_m.predict(train_images_01, train_labels_01)
# # print('Training Accuracy',accuracy_score(y_train, train_labels_01))
y_test = log_reg_m.predict(test_images_01, test_labels_01)
print('Test Accuracy',accuracy_score(y_test, test_labels_01))
y_test = log_reg_m.predict_alg1(test_images_01, test_labels_01, 5)
print('Test Accuracy',accuracy_score(y_test, test_labels_01))
# accuracy, precision, recall, f_measure = log_reg_m.performance(y_train, train_labels_01)
# print(accuracy, precision, recall, f_measure)
# accuracy, precision, recall, f_measure = log_reg_m.performance(y_test, test_labels_01)
# print(accuracy, precision, recall, f_measure)

plt.subplot(2, 1, 1)
plt.plot(costs)
plt.subplot(2, 1, 2)
plt.plot(costs_distributed)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title('Cost reduction over time')
plt.show()







print("finished")

