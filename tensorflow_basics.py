
# %tensorflow_version 2.x
import tensorflow as tf
import os
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


train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64, shuffle=False)

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
    x = tf.pad(x, pad)
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
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


# three_layer_convnet_test()

def training_step(model_fn, x, y, params, learning_rate):
    with tf.GradientTape() as tape:
        scores = model_fn(x, params)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        total_loss = tf.reduce_mean(loss)
        grad_params = tape.gradient(total_loss, params)
        for w, grad_w in zip(params, grad_params):
            w.assign_sub(learning_rate * grad_w)
        return total_loss

def check_accuracy(dset, x, model_fn, params):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        scores_np = model_fn(x_batch, params).numpy()
        y_pred = scores_np.argmax(axis=1)
        num_samples+= x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct)/num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def train_part2(model_fn, init_fn, learning_rate):
    params = init_fn()
    for t, (x_np, y_np) in enumerate(train_dset):
        loss = training_step(model_fn, x_np, y_np, params, learning_rate)
        if t%print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss))
            check_accuracy(val_dset, x_np, model_fn, params)

def random_weight(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.keras.backend.random_normal(shape) * np.sqrt(2.0/fan_in)

def two_layer_fc_init():
    hidden_layer_size = 4000
    w1 = tf.Variable(random_weight((3*32*32, hidden_layer_size)))
    w2 = tf.Variable(random_weight((hidden_layer_size, 10)))
    return [w1, w2]

learning_rate = 3e-3
# train_part2(two_layer_fc, two_layer_fc_init, learning_rate)

def three_layer_convnet_init():
    conv_w1 = tf.Variable(random_weight((5,5,3,32)))
    conv_b1 = tf.Variable(np.zeros((32,)), dtype=tf.float32)
    conv_w2 = tf.Variable(random_weight((3, 3, 32, 16)))
    conv_b2 = tf.Variable(np.zeros((16,)), dtype=tf.float32)
    fc_w = tf.Variable(random_weight((32*32*16, 10)))
    fc_b = tf.Variable(np.zeros((10,)), dtype=tf.float32)
    return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]

# train_part2(three_layer_convnet, three_layer_convnet_init, learning_rate)

class TwoLayerFC(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super(TwoLayerFC, self).__init__()
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu', use_bias=True, kernel_initializer=initializer)
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='relu', use_bias=True, kernel_initializer=initializer)
        self.flatten = tf.keras.layers.Flatten()

    def __call__(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.flatten(x)
        return x

def TwoLayerFC_test():
    input_size, hidden_size, num_classes = 50, 42, 10
    x = tf.zeros((64, input_size))
    model = TwoLayerFC(hidden_size, num_classes)
    scores = model(x)
    with tf.device(device):
        print(scores.shape)

# TwoLayerFC_test()

class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super(ThreeLayerConvNet, self).__init__()
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        self.conv1 = tf.keras.layers.Conv2D(channel_1, (5,5), padding='valid', kernel_initializer=initializer, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(channel_2, (3,3), padding='valid', kernel_initializer=initializer, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer, activation='softmax')

    def __call__(self, x, training=False):
        pad = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        x = tf.pad(x, pad)
        x = self.conv1(x)
        pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.pad(x, pad)
        x = self.conv2(x)
        x = self.flatten(x)
        scores = self.fc(x)
        return scores

def test_ThreeLayerConvNet():
    channel_1, channel_2, num_classes = 12, 8, 10
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    with tf.device(device):
        x = tf.zeros((64, 3, 32, 32))
        scores = model(x)
        print(scores.shape)

# test_ThreeLayerConvNet()


def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1, is_training=False):
    with tf.device(device):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model = model_init_fn()
        optimizer = optimizer_init_fn()
        train_loss =tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        t = 0
        for epoch in range(num_epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            for x_np, y_np in train_dset:
                with tf.GradientTape() as tape:
                    scores = model(x_np, training=is_training)
                    loss = loss_fn(y_np, scores)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_np, scores)

                    if t%print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for x_test, y_test in val_dset:
                            pred = model(x_test, training=False)
                            test_loss = loss_fn(y_test, pred)
                            val_loss.update_state(test_loss)
                            val_accuracy.update_state(y_np, scores)
                        template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
                        print(template.format(t, epoch + 1,
                                              train_loss.result(),
                                              train_accuracy.result() * 100,
                                              val_loss.result(),
                                              val_accuracy.result() * 100))
                    t += 1


hidden_size, num_classes = 512, 10
learning_rate = 1e-2

def model_init_fn():
    return TwoLayerFC(hidden_size, num_classes)

def optimizer_init_fn():
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_part34(model_init_fn, optimizer_init_fn, num_epochs=1)

