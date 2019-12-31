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

def conv_forward_naive_m(x, w, b, conv_params):
    out, cache = None, None
    N, C, W, H = x.shape # #images, #channels, width, height
    K, _, F, F = w.shape # #filters, #channels, field width, field height
    P = conv_params['padding']
    S = conv_params['stride']
    W_out = int((W - F + 2 * P) / S + 1)
    H_out = int((H - F + 2 * P) / S + 1)
    C_out = K
    out_shape = (N, C_out, W_out, H_out)
    out = np.zeros(out_shape)
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant', constant_values=0)
    for k in range(K): #over filters
        for ix in range(W_out):
            for iy in range(H_out):
                out[:,k,ix,iy] = np.sum(w[k,np.arange(C)]*x_pad[np.ix_(np.arange(N), np.arange(C), np.arange(ix*S,ix*S+F),np.arange(iy*S,iy*S+F))], axis=(1,2,3)) + b[k]
    # for n in range(N): #over images
    #     for k in range(K): #over filters
    #         for c in range(C): #over channels in image
    #             for ix in range(W_out):
    #                 for iy in range(H_out):
    #                     # print(n, k, ix, iy)
    #                     # print(list(np.arange(ix * S, ix * S + F)))
    #                     # print(list(np.arange(iy * S, iy * S + F)))
    #                     out[n,k,ix,iy] = out[n,k,ix,iy] + np.sum(w[k,c]*x[n,c][np.ix_(np.arange(ix*S,ix*S+F).tolist(),np.arange(iy*S,iy*S+F).tolist())])
    cache = (x, w, b, conv_params)

    return out, cache

def conv_backward_naive_m(self, dout, cache):
    x, w, b, conv_params = cache
    N, K, W_out, H_out = dout.shape  # #images, #filters, out width, out height
    _, C, F, _ = w.shape  # #filters, #channels, receptive field width, receptive field height
    P = conv_params['padding']
    S = conv_params['stride']
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant', constant_values=0)
    dx_pad = np.zeros_like(x_pad)#images, #channels, width, height
    dw = np.zeros_like(w)#filters, #channels, field width, field height
    db = np.zeros_like(b)

    for n in range(N): #over images
        for k in range(K): #over filters
            for ix in range(W_out):
                for iy in range(H_out):
                    dw[k,:,:,:] += dout[n,k,ix,iy]*x_pad[n][np.ix_(np.arange(C), np.arange(ix*S,ix*S+F),np.arange(iy*S,iy*S+F))] #equal to convolution X and dL/dO
                    db[k] += dout[n,k,ix,iy]
                    dx_pad[n][np.ix_(np.arange(C), np.arange(ix*S,ix*S+F),np.arange(iy*S,iy*S+F))] += w[k,:,:,:] * dout[n,k,ix,iy] #equal to full convolution with W inverted
    #https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
    dx = dx_pad[:, :, P:-P, P:-P]

    return dx, dw, db

def max_pool_forward_naive_m(self, x, pool_param):
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    S = pool_param["stride"]
    N, C, H, W = x.shape
    H_out = int((H - PH) / S + 1)
    W_out = int((W - PW) / S + 1)
    out_shape = (N, C, H_out, W_out)
    out = np.zeros(out_shape)
    for n in range(N): #over images
        for ix in range(W_out):
            for iy in range(H_out):
                out[n,:,ix,iy] = np.amax(x[n][np.ix_(np.arange(C),np.arange(ix*S,ix*S+PH),np.arange(iy*S,iy*S+PW))], axis=(1,2))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward_m(self, dout, cache):
    x, pool_param = cache
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]
    S = pool_param["stride"]
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    dx = np.zeros_like(x)
    for n in range(N): #over images
        for c in range(C):
            for ix in range(W_out):
                for iy in range(H_out):
                    max_index = np.unravel_index((x[n,c][np.ix_(np.arange(ix*S,ix*S+PH),np.arange(iy*S,iy*S+PW))]).argmax(), \
                                                 (x[n,c][np.ix_(np.arange(ix*S,ix*S+PH),np.arange(iy*S,iy*S+PW))]).shape)
                    max_index = np.array(max_index)+np.array((ix*S,iy*S))
                    dx[n,c,max_index[0],max_index[1]] = dout[n, c, ix, iy]

    return dx

def conv_relu_pool_forward_m(x, w, b, conv_param, pool_param):
    z_conv, cache_conv = conv_forward_naive_m(x, w, b, conv_param)
    z_relu, cache_relu = relu_forward_m(z_conv)
    z_pool, cache_pool = max_pool_forward_naive_m(z_relu, pool_param)
    cache = (cache_conv, cache_relu, cache_pool)

    return z_pool, cache

def conv_relu_pool_backward_m(self, dout, cache):
    cache_conv, cache_relu, cache_pool = cache
    dout_pool = max_pool_backward_m(dout, cache_pool)
    dout_relu = relu_backward_m(dout_pool, cache_relu)
    dx, dw, db = conv_backward_naive_m(dout_relu, cache_conv)

    return dx, dw, db