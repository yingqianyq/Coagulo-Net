import tensorflow as tf
import numpy as np


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class PINN(tf.keras.Model):

    def __init__(self, name="pinn"):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(3)
            ]
        )
        self.nn.build(input_shape=[None, 1])
        self.a = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.c = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.d = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam()

        self._name = name
    
    def call(self, t):
        return self.nn.call(t)

    @tf.function
    def ode(self, t):
        u = self.call(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        return tf.concat([eq1, eq2, eq3], axis=-1)

    def loss_function(self, t_f, f, t_u, u):
        f_pred = self.ode(t_f)
        u_pred = self.call(t_u)
        loss_f = tf.reduce_mean((f_pred - f)**2)
        loss_u = tf.reduce_mean((u_pred - u)**2)
        return loss_f + loss_u

    def train_op(self, t_f, f, t_u, u):
        with tf.GradientTape() as tape:
            f_pred = self.ode(t_f)
            u_pred = self.call(t_u)
            loss_f = tf.reduce_mean((f_pred - f)**2)
            loss_u = tf.reduce_mean((u_pred - u)**2)
            loss = loss_u + loss_f
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t_f, f, t_u, u, niter=10000):
        t_f = tf.constant(t_f, tf.float32)
        f = tf.constant(f, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)
        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda: tf.reduce_mean((self.ode(t_f) - f)**2) + tf.reduce_mean((self.call(t_u) - u)**2)
        )

        loss = []
        min_loss = 10000
        for it in range(niter):
            loss += [train_op(t_f, f, t_u, u).numpy()]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class Meta(tf.keras.Model):

    def __init__(self, name="meta"):
        super().__init__()

        self.nn_1 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(3)
            ]
        )
        self.nn_2 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(3)
            ]
        )
        self.nn_1.build(input_shape=[None, 1])
        self.nn_2.build(input_shape=[None, 1])

        self.a = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.c = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.d = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam()

        self._name = name
    
    @tf.function
    def ode(self, t, nn):
        u = nn(t)
        v = tf.ones_like(t)
        u_t = jvp(u, t, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        return tf.concat([eq1, eq2, eq3], axis=-1)

    def loss_function(self, t_f, f, t_u, u, nn):
        f_pred = self.ode(t_f, nn)
        u_pred = nn(t_u)
        loss_f = tf.reduce_mean((f_pred - f)**2)
        loss_u = tf.reduce_mean((u_pred - u)**2)
        return loss_f + loss_u

    def train_op(
        self,
        t1_f, 
        f1, 
        t1_u, 
        u1,
        t2_f,
        f2,
        t2_u,
        u2,
    ):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t1_f, f1, t1_u, u1, self.nn_1) + \
                   self.loss_function(t2_f, f2, t2_u, u2, self.nn_2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2, niter=10000):
        t1_f = tf.constant(t1_f, tf.float32)
        f1 = tf.constant(f1, tf.float32)
        t1_u = tf.constant(t1_u, tf.float32)
        u1 = tf.constant(u1, tf.float32)

        t2_f = tf.constant(t2_f, tf.float32)
        f2 = tf.constant(f2, tf.float32)
        t2_u = tf.constant(t2_u, tf.float32)
        u2 = tf.constant(u2, tf.float32)

        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda: self.loss_function(t1_f, f1, t1_u, u1, self.nn_1) + self.loss_function(t2_f, f2, t2_u, u2, self.nn_2)
        )

        loss = []
        min_loss = 10000
        for it in range(niter):
            loss += [
                train_op(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2).numpy()
            ]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)


class Meta2(tf.keras.Model):

    def __init__(self, name="meta2"):
        super().__init__()

        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(50, activation=tf.tanh),
                tf.keras.layers.Dense(6)
            ]
        )
        self.nn.build(input_shape=[None, 1])

        self.a = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.c = tf.Variable(0.0, dtype=tf.float32, trainable=True)
        self.d = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        self.opt = tf.keras.optimizers.Adam()

        self._name = name
    
    @tf.function
    def ode(self, t1, t2):
        # the first head
        u = self.nn(t1)[:, 0:3]
        v = tf.ones_like(t1)
        u_t = jvp(u, t1, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        f1 = tf.concat([eq1, eq2, eq3], axis=-1)

        # the second head
        u = self.nn(t2)[:, 3:6]
        v = tf.ones_like(t2)
        u_t = jvp(u, t2, v)[0]
        u1_t, u2_t, u3_t = tf.split(u_t, 3, axis=-1)
        u1, u2, u3 = tf.split(u, 3, axis=-1)
        eq1 = u1_t - self.a * u1 * u3
        eq2 = u2_t - self.b * u2 * u3
        eq3 = u3_t - self.c * u1**2 - self.d * u2**2
        f2 = tf.concat([eq1, eq2, eq3], axis=-1)
        return f1, f2

    def loss_function(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2):
        f1_pred, f2_pred = self.ode(t1_f, t2_f)
        u1_pred = self.nn(t1_f)[:, 0:3]
        u2_pred = self.nn(t2_f)[:, 3:6]
        loss_f = tf.reduce_mean((f1_pred - f1)**2) + tf.reduce_mean((f2_pred - f2)**2)
        loss_u = tf.reduce_mean((u1_pred - u1)**2) + tf.reduce_mean((u2_pred - u2)**2)
        return loss_f + loss_u

    def train_op(
        self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2,
    ):
        with tf.GradientTape() as tape:
            loss = self.loss_function(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2, niter=10000):
        t1_f = tf.constant(t1_f, tf.float32)
        f1 = tf.constant(f1, tf.float32)
        t1_u = tf.constant(t1_u, tf.float32)
        u1 = tf.constant(u1, tf.float32)

        t2_f = tf.constant(t2_f, tf.float32)
        f2 = tf.constant(f2, tf.float32)
        t2_u = tf.constant(t2_u, tf.float32)
        u2 = tf.constant(u2, tf.float32)

        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda: self.loss_function(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2,)
        )

        loss = []
        min_loss = 10000
        for it in range(niter):
            loss += [
                train_op(t1_f, f1, t1_u, u1, t2_f, f2, t2_u, u2).numpy()
            ]
            if it % 1000 == 0:
                current_loss = loss_op().numpy()
                print(it, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name,
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights("./checkpoints/"+self.name)
