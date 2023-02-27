# known: all data + parameters except kapc
# unknown: kapc
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
import models_tf as models

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--lam', type=float)
parser.add_argument('--output', type=str)
args = parser.parse_args()
print(args)

data = sio.loadmat("../solution.mat")
c0 = data["IC"]
t = data["t"]
c_ref = data["y"]

t_train = t

scale = np.array(
    [10, 1, 100, 100, 0.001, 10, 0.1, 1000]
).reshape([-1])
c_ref = c_ref / scale
c0 = c0 / scale

model = models.PINN(
    c0=c0,
    output_dim=8,
    scale=scale,
    units=100, 
    activation=tf.tanh,
    eps=1e-4,
)

t_ode = t_train[::10]
t_u = t_train
u = c_ref
loss, data_loss, ode_loss = model.train(t_ode, t_u, u, niter=1000000, lam=args.lam)

model.restore()
c_pred = model.call(
    tf.constant(t_train, tf.float32),
)

# k_apc = tf.math.exp(model.log_k_apc)
# print(k_apc)
k8 = tf.math.exp(model.log_k8)
print(k8)
ka = tf.math.exp(model.log_ka)
print(ka)
k5 = tf.math.exp(model.log_k5)
print(k5)
h5 = tf.math.exp(model.log_h5)
print(h5)


sio.savemat(args.output, {"k8": k8.numpy(), "ka": ka.numpy(), "k5": k5.numpy(), "h5": h5.numpy()})