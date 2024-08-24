#!/usr/bin/env python
# coding: utf-8

# # Inverse problem
# known: all data, but sparse  
# unknown: k8, ka, k5, h5  
# results record: https://outlookuga-my.sharepoint.com/:x:/r/personal/yq88347_uga_edu/_layouts/15/Doc.aspx?sourcedoc=%7Be6449b96-7895-43ff-a1e4-42fc2118f900%7D&action=editnew

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
import models_tf as models

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--datainput', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()
print(args)

data = sio.loadmat("../../data/" + args.datainput)
c0 = data["IC"]
t = data["t"]
c_ref = data["y"]

t_train = t

scale = np.array(
    [10, 1, 100, 100, 0.0001, 10, 0.1, 1000]
).reshape([-1])
c_ref = c_ref / scale
c0 = c0 / scale

model = models.PINN(
    c0=c0,
    output_dim=8,
    scale=scale,
    units=100, 
    activation=tf.tanh,
    eps=1e-5,
)

# t_ode = t_train[::10]
# t_u = t_train
# u = c_ref
t_ode = t_train[:201]
t_u = t_train
u = c_ref
loss, ode_loss, data_loss, min_loss, h5_list = model.train(t_ode, t_u, u, niter=500000)

model.restore()
c_pred = model.call(
    tf.constant(t_train, tf.float32),
)

L2 = np.sqrt(np.sum((c_pred - c_ref) ** 2, axis=0) / np.sum(c_ref ** 2, axis=0))

param_inf = []
# ka = 1.2
param_inf.append(str(tf.math.exp(model.log_ka).numpy()))

# k8 = 0.00001
param_inf.append(str(tf.math.exp(model.log_k8).numpy()))

# k5 = 0.17
param_inf.append(str(tf.math.exp(model.log_k5).numpy()))

# h5 = 0.31
param_inf.append(str(tf.math.exp(model.log_h5).numpy()))

print(param_inf)

with open(args.output, 'a') as f:
    f.write(','.join(param_inf) + '\n')
