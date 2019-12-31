import numpy as np

def sgd(w, dw, config):
    lr = config["learning_rate"]
    w -= lr * dw
    return w


def sgd_momentum(w, dw, name, config):
    if v.get(name) is None:
        v[name] = np.zeros_like(w)
    mu = config.get("momentum")
    lr = config.get("learning_rate")
    v[name] = mu * v[name] - lr * dw
    w += v[name]
    return w


def nesterov_momentum(w, dw, name, config):
    if v.get(name) is None:
        v[name] = np.zeros_like(w)
    mu = config.get("momentum")
    lr = config.get("learning_rate")
    dw_ahead = dw + mu * v[name]
    v[name] = mu * v[name] - lr * dw_ahead
    w += v[name]
    return w


def rmsprop(w, dw, name, config):
    if v.get(name) is None:
        v[name] = np.zeros_like(w)
    lr = config.get("learning_rate")
    dr = config.get("decay_rate")
    v[name] = dr * v[name] + (1 - dr) * dw ** 2
    w = w - lr * dw / (np.sqrt(v[name]) + 1e-8)
    return w


def adam(w, dw, name, config):
    if config.get(name + "m") is None:
        config.setdefault(name + "m", np.zeros_like(w))
        config.setdefault(name + "v", np.zeros_like(w))
        config.setdefault(name + "t", 0)

    next_w = 0
    config[name + "t"] += 1
    config[name + "m"] = config["beta1"] * config[name + "m"] + (1 - config["beta1"]) * dw
    mt = config[name + "m"] / (1 - config['beta1'] ** config[name + "t"])
    config[name + "v"] = config["beta2"] * config[name + "v"] + (1 - config["beta2"]) * (dw ** 2)
    vt = config[name + "v"] / (1 - config['beta2'] ** config[name + "t"])
    w = w - config["learning_rate"] * mt / (np.sqrt(vt) + config["epsilon"])

    return w, config