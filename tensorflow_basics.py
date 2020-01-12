import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

class Dataset:
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))


# train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
# val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
# test_dset = Dataset(X_test, y_test, batch_size=64, shuffle=False)

# for t, (x, y) in enumerate(train_dset):
#     print(t, x.shape, y.shape)
#     if t > 5:
#         break

USE_GPU = False
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

print_every = 100

print('Using device', device)

def flatten(x):
    N = tf.shape(x)[0]
    return tf.reshape(x, (N, -1))

def test_flatten():
    x_np = np.arange(24).reshape((2,3,4))
    print('x_np:\n', x_np)
    x_flat_np = flatten(x_np)
    print('x_flat_np:\n', x_flat_np)

# test_flatten()

def two_layer_fc(x, params):
    w1, w2 = params
    x = flatten(x)
    h = tf.nn.relu(tf.matmul(x, w1))
    scores = tf.matmul(h, w2)
    return scores

def two_layer_fc_test():
    hidden_layer_size = 42
    with tf.device(device):
        x = tf.zeros((64, 32, 32, 3))
        w1 = tf.zeros((32,32,3, hidden_layer_size))
        w2 = tf.zeros((hidden_layer_size, 10))
        scores = two_layer_fc(x, [w1, w2])
        print(scores.shape)

# two_layer_fc_test()

def three_layer_convnet(x, params):
    conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b = params
    scores = None
    pad = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    x_pad = tf.pad(x, pad)
    conv1 = tf.nn.conv2d(x_pad, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
    relu1 = tf.nn.relu(conv1)
    pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    relu1 = tf.pad(relu1, pad)
    conv2 = tf.nn.conv2d(relu1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    relu2 = tf.nn.relu(conv2)
    relu2 = flatten(relu2)
    scores = tf.matmul(relu2, fc_w) + fc_b
    return scores


def three_layer_convnet_test():
    with tf.device(device):
        x = tf.zeros((64, 32, 32, 3))
        conv_w1 = tf.zeros((5, 5, 3, 6))
        conv_b1 = tf.zeros((6,))
        conv_w2 = tf.zeros((3, 3, 6, 9))
        conv_b2 = tf.zeros((9,))
        fc_w = tf.zeros((32 * 32 * 9, 10))
        fc_b = tf.zeros((10,))
        params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
        scores = three_layer_convnet(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape
    # [batch_size, height, width, channels]
    print('scores_np has shape: ', scores.shape)


three_layer_convnet_test()

