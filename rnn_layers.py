# from __future__ import print_function, division
# from builtins import range
import numpy as np

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + h
        pos = f(x)
        x[ix] = old - h
        neg = f(x)
        x[ix] = old

        grad[ix] = np.sum((pos-neg)*df)/(2*h)
        it.iternext()
    return grad

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = None, None
    out = x@Wx + prev_h@Wh + b
    next_h = np.tanh(out)
    cache = (x, Wx, Wh, prev_h, out)

    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    x, Wx, Wh, prev_h, out = cache
    dx, dWx, dWh, dprev_h, db = None, None, None, None, None
    dnext_h = dnext_h*(1-np.tanh(out)**2)
    db = np.sum(dnext_h,axis = 0)
    dWh = (prev_h.T).dot(dnext_h)
    dprev_h = dnext_h.dot(Wh.T)
    dWx = (x.T).dot(dnext_h)
    dx = dnext_h.dot(Wx.T)
    return dx, dprev_h, dWx, dWh, db


def word_embedding_forward(x, W):
    out = W[x]
    cache = (x, W)
    return out, cache

def word_embedding_backward(dout, cache):
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW

def temporal_affine_forward(x, w, b):
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N*T, D)@w).reshape(N,T,M) + b
    cache = (x, w, b, out)
    return out, cache

def temporal_affine_backward(dout, cache):
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]
    dx = dout.reshape(N*T,M).dot(w.T).reshape(N,T,D)
    dw = (dout.reshape(N*T, M).T@x.reshape(N*T, D)).T
    db = dout.sum(axis=(0,1))

    return dx, dw, db

def temporal_softmax_loss(x, y, mask):
    N: object
    N, T, V = x.shape
    x_flat = x.reshape(N*T, V)
    y_flat = y.reshape(N*T)
    mask_flat = mask.reshape(N*T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat*np.log(probs[np.arange(N*T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N*T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    dx = dx_flat.reshape(N, T, V)

    return loss, dx

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N,T,H))
    cache = []
    h[:, 0, :] = h0
    for t in range(T):
        if t == 0:
            h[:, t, :], cache_t = rnn_step_forward(x[:, t, :], h0, Wx, Wh, b)
        else:
            h[:, t, :], cache_t = rnn_step_forward(x[:, t, :], h[:, t-1, :], Wx, Wh, b)
        cache.append(cache_t)

    return h, cache

def rnn_backward(dh, cache):
    dxi, dprev_hi, dWxi, dWhi, dbi = None, None, None, None, None
    N, T, H = dh.shape
    dxl, dprev_h, dWx, dWh, db = rnn_step_backward(dh[:, T-1, :], cache[T-1])
    dprev_hi = dprev_h
    _, D = dxl.shape
    dx = np.zeros((N, T, D))
    dx[:, T - 1, :] = dxl
    for t in reversed(range(T-1)):
        dxi, dprev_hi, dWxi, dWhi, dbi = rnn_step_backward(dh[:, t, :] + dprev_hi, cache[t])
        dx[:, t, :] = dxi
        dWx += dWxi
        dWh += dWhi
        db += dbi
    dh0 = dprev_hi

    return dx, dh0, dWx, dWh, db



