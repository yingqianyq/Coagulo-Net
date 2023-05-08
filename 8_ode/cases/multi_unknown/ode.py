#!/usr/bin/env python
# coding: utf-8

# # Inverse problem
# known: all data  
# unknown: k8, ka, k5, h5

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
import models_tf as models

# data = sio.loadmat("../test.mat")
data = sio.loadmat("../data.mat")
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
loss, data_loss, ode_loss = model.train(t_ode, t_u, u, niter=500000)

model.restore()
c_pred = model.call(
    tf.constant(t_train, tf.float32),
)

L2 = np.sqrt(np.sum((c_pred - c_ref) ** 2, axis=0) / np.sum(c_ref ** 2, axis=0))
print("L2:", L2)

# ka = 1.2
print("ka: 1.2, ", tf.math.exp(model.log_ka))

# k8 = 0.00001
print("k8: 0.00001, ", tf.math.exp(model.log_k8))

# k5 = 0.17
print("k5: 0.17, ", tf.math.exp(model.log_k5))

# h5 = 0.31
print("h5: 0.31, ", tf.math.exp(model.log_h5))
