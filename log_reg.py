import mnist
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression

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
    def __init__(self, n_features, lr, num_iter):
        self.w = np.zeros(1, n_features)
        self.b = 0

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def calc_grad(self, X, Y):
        m = X.shape[0]

        final_result = self.sigmoid(np.dot(self.w,X.T) + self.b)
        Y_T = Y.T
        cost = (-1/m)*(np.sum(Y_T*np.log(final_result)) + ((1 - Y_T)*(np.log(1-final_result))))

        self.dw = (1/m)*(np.dot(X.T, (final_result - Y.T).T))
        self.db = (1/m)*(np.sum(final_result - Y.T))

        # grads = {"dw" : dw, "db" : db}

        return cost

    def model_predict(self, X_train, y_train):
        costs = []
        for i in range(self.num_iter):
            grad, cost = self.calc_grad(X_train, y_train)
            self.dw = grad["dw"]
            self.db = grad["db"]
            self.w = self.w - self.lr * self.dw.T
            self.b = self.b - self.lr * self.db

            if(i%128 == 0):
                costs.append(cost)

        coeff = {"w" : self.w, "b" : self.b}
        gradient = {"dw" : self.dw, "db" : self.db}

        return coeff, gradient, costs

    def predict(self, final_pred, m):
        y_pred = np.zeros((1,m))
        for i in range(final_pred.shape[1]):
            if( final_pred[0][i] > 0.5):
                y_pred[0][i] = 1
        return y_pred

    def fit(self, X_train, y_train):
        n_features = X_train.shape[1]
        print('Number of Features', n_features)
        w, b = self.weight_initialization(n_features)
        coeff, gradient, costs = self.model_predict(X_train, y_train)
        w = coeff["w"]
        b = coeff["b"]
        final_train_pred = self.sigmoid(np.dot(w, X_train.T) + b)
        # final_test_pred = self.
        y_pred = self.predict(final_train_pred, X_train.shape[0])
        return self


log_reg_m = Log_Reg(lr=0.0001, num_iter=100)
log_reg_m.fit(X_train=train_images_01, y_train=train_labels_01)








print("finished")

