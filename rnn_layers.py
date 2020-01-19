from __future__ import print_function, division
from builtins import range
import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = None, None
    out = x@Wx + prev_h@Wh + b
    next_h = np.tanh(out)
    cache = (x, Wx, Wh, prev_h, next_h, out)

    return next_h, cache