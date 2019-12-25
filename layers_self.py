import numpy as np

def affine_forward_m(x, w, b):
    z = x @ w + b  # z = WX + b
    cache = (x, w, b)
    return z, cache

def affine_backward_m(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    X = np.reshape(x, (N, -1))  # dout=dLoss/dz - gradient from previous step
    dx = dout @ w.T  # dz/dx = dot(W.T, dout)
    dw = X.T @ dout  # dz/wd = dot(dout, X)
    db = np.sum(dout, axis=0)  # dz/db = sum(dout)
    return dx, dw, db


def relu_forward_m(x):
    cache = x
    out = x * (x > 0)  # ReLu
    return out, cache


def relu_backward_m(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def softmax_loss_m(x, y):
    N = x.shape[0]
    shiftx = x.T - x.T.max(0)
    shiftx = np.exp(shiftx)
    out = shiftx / np.sum(shiftx, axis=0)
    out = out.T
    loss = np.log(out[(np.arange(N), y)] / np.sum(out, axis=1) + 1e-8)
    loss = -np.sum(loss) / N
    dout = out.copy()  # dLoss/dx
    dout[(np.arange(N), y)] -= 1
    dout /= N
    return loss, dout


def affine_relu_forward_m(x, w, b):
    z_fc, cache_fc = affine_forward_m(x, w, b)
    z_relu, cache_relu = relu_forward_m(z_fc)
    cache = (cache_fc, cache_relu)
    return z_relu, cache


def affine_relu_backward_m(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward_m(dout, relu_cache)
    dx, dw, db = affine_backward_m(da, fc_cache)
    return dx, dw, db


def affine_batchnorm_relu_forward_m(x, w, b, gamma, betta, mode, name):
    z_fc, cache_fc = affine_forward_m(x, w, b)
    z_bn, cache_bn = batch_norm_forward_m(z_fc, gamma, betta, mode, name)
    z_relu, cache_relu = relu_forward_m(z_bn)
    cache = (cache_fc, cache_bn, cache_relu)
    return z_relu, cache


def affine_batchnorm_relu_backward_m(dout, cache):
    fc_cache, cache_bn, relu_cache = cache
    da = relu_backward_m(dout, relu_cache)
    dbn, dgamma, dbetta = batch_norm_backward_m(da, cache_bn)
    dx, dw, db = affine_backward_m(dbn, fc_cache)
    return dx, dw, db, dgamma, dbetta


def batch_norm_forward_m(x, gamma, betta, mode, name, bn_param):
    cache = None
    if bn_param.get('running_mean' + name) is None:
       bn_param['running_mean' + name] = np.zeros_like(x.shape[0])
       bn_param['running_var' + name] = np.zeros_like(x.shape[0])

    running_mean = bn_param.get('running_mean' + name)
    running_var = bn_param.get('running_var' + name)
    if mode == "train":
        mu = np.sum(x, axis=0) / x.shape[0]
        sigma = np.sum(np.power(x - mu, 2), axis=0) / x.shape[0]
        x_bn = (x - mu) / np.sqrt(sigma + 1e-8)
        y = gamma * x_bn + betta
        running_mean = 0.9 * running_mean + (1 - 0.9) * mu
        running_var = 0.9 * running_var + (1 - 0.9) * sigma
        cache = (x, x_bn, mu, sigma, gamma, betta)
    else:
        x_bn = (x - running_mean) / np.sqrt(running_var + 1e-8)
        y = gamma * x_bn + betta
    bn_param['running_mean' + name] = running_mean
    bn_param['running_var' + name] = running_var

    return y, cache


def dropout_forward_m(x, p, mode, cache):
    r = np.random.binomial(1, p, x.shape) / p
    if mode == 'train':
        y = r * x
    else:
        y = x
    fc_cache, cache_bn, relu_cache = cache
    cache_do = (x, r)
    cache_out = (fc_cache, cache_bn, relu_cache, cache_do)
    return y, cache_out


def dropout_backward_m(dout, cache):
    fc_cache, cache_bn, relu_cache, cache_do = cache
    x, r = cache_do
    cache_out = (fc_cache, cache_bn, relu_cache)
    return dout * r, cache_out


def batch_norm_backward_m(dout, cache):
    x, x_bn, mu, sigma, gamma, betta = cache
    m = x.shape[0]
    dx_bn = dout * gamma
    dgamma = np.sum(dout * x_bn, axis=0)
    dbetta = np.sum(dout, axis=0)
    dsigma = np.sum(dout * (x - mu), axis=0) * -0.5 * gamma * np.power(sigma + 1e-8, -3 / 2)
    dmu = np.sum(dout * -gamma, axis=0) / np.sqrt(sigma + 1e-8) + \
          dsigma * -2 * np.sum(x - mu, axis=0) / m
    dx = dx_bn / np.sqrt(sigma + 1e-8) + dsigma * 2 * (x - mu) / m + dmu / m
    return dx, dgamma, dbetta