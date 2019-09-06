import mnist
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
print(train_labels_01.shape )
print(train_images_01.shape )
print(test_labels_01.shape )
print(test_images_01.shape )

class Log_Reg:
    def __init__(self, lr, num_iter):
        self.lr = lr
        self.num_iter = num_iter

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
            self.w = self.w - self.lr * self.dw.T
            self.b = self.b - self.lr * self.db
            self.costs[i] = cost
        return self.costs

    def predict(self, X_test, y_test):
        y_T = y_test.T
        X_T = X_test.T
        y_pred = self.sigmoid(np.dot(self.w, X_T) + self.b)
        y_pred_bin = np.zeros(y_pred.shape[1])
        for i in range(y_pred.shape[1]):
            if (y_pred[0][i] > 0.5):
                y_pred_bin[i] = 1
        return y_pred_bin


log_reg_m = Log_Reg(lr=0.0001, num_iter=100)
costs = log_reg_m.fit(X_train=train_images_01, y_train=train_labels_01)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title('Cost reduction over time')
plt.show()
y_train = log_reg_m.predict(train_images_01, train_labels_01)
print('Training Accuracy',accuracy_score(y_train, train_labels_01))
#
y_test = log_reg_m.predict(test_images_01, test_labels_01)
print('Test Accuracy',accuracy_score(y_test, test_labels_01))







print("finished")

