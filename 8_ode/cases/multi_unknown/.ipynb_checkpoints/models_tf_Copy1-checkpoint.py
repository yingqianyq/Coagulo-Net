import tensorflow as tf


def jvp(y, x, v):
    # For more information, see https://github.com/renmengye/tensorflow-forward-ad/issues/2
    u = tf.ones_like(y)  # unimportant
    g = tf.gradients(y, x, grad_ys=u)
    return tf.gradients(g, u, grad_ys=v)


class NN(tf.keras.Model):

    def __init__(self, c0, scale, output_dim, units=100, activation=tf.tanh, name="nn"):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(output_dim)
            ]
        )
        self.activation = activation
        self.nn.build(input_shape=[None, 1])
        self.opt = tf.keras.optimizers.Adam()

        # self.c0 = tf.constant(c0, tf.float32)
        # self.scale = tf.constant(scale, tf.float32)
        self.c0 = tf.Variable(c0, dtype=tf.float32, trainable=False)
        # self.scale = tf.Variable(scale, dtype=tf.float32, trainable=False)

        self._name = name
    
    def call(self, inputs):
        # _inputs = tf.math.exp(inputs)
        # output = self.c0 * tf.math.exp(inputs * self.nn.call(inputs))
        # output = tf.math.exp(self.nn.call(inputs))
        # output = self.c0 + inputs * self.nn.call(inputs)
        output = self.nn.call(inputs)
        return output

    def train_op(self, inputs, targets):
        with tf.GradientTape() as tape:
            _loss = tf.reduce_mean((self.call(inputs)-targets)**2, axis=0)
            loss = tf.reduce_sum(_loss)
            # constants = tf.stop_gradient(_loss)
            # loss = tf.reduce_sum(loss / tf.stop_gradient(loss))
            # loss = tf.reduce_sum(_loss / constants)
            # loss = tf.reduce_mean(loss / self.scale ** 2)

        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return tf.reduce_sum(_loss)

    def train(self, inputs, targets, niter=20000):
        inputs = tf.constant(inputs, tf.float32)
        targets = tf.constant(targets, tf.float32)
        train_op = tf.function(self.train_op)
        loss_op = tf.function(
            lambda : tf.reduce_mean((self.call(inputs)-targets)**2),
        )

        loss = []
        min_loss = 10000000
        for i in range(niter):
            loss_value = train_op(inputs, targets)
            loss += [loss_value.numpy()]
            if i % 1000 == 0:
                current_loss = loss_op().numpy()
                print(i, current_loss)
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name, 
                        overwrite=True,
                    )
        return loss

    def restore(self):
        self.load_weights(filepath="./checkpoints/"+self.name)


class PINN(tf.keras.Model):

    def __init__(self, c0, scale, output_dim, units=100, activation=tf.tanh, eps=1, name="pinn"):
        super().__init__()
        self.nn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(units, activation=activation),
                tf.keras.layers.Dense(output_dim)
            ]
        )
        self.c0 = tf.constant(c0, tf.float32)
        self.scale = tf.constant(scale, tf.float32)
        self.activation = activation
        self.eps = tf.constant(eps, tf.float32)
        self.nn.build(input_shape=[None, 1])

        print("eps: ", self.eps)
        
        # self.log_k1 = tf.Variable(0.0, dtype=tf.float32, name="log_k1")
        # self.log_k_apc = tf.Variable(0.0, dtype=tf.float32, name="log_k_apc")
        self.log_k8 = tf.Variable(0.0, dtype=tf.float32, name="log_k8")
        self.log_ka = tf.Variable(0.0, dtype=tf.float32, name="log_ka")
        self.log_k5 = tf.Variable(0.0, dtype=tf.float32, name="log_k5")
        self.log_h5 = tf.Variable(0.0, dtype=tf.float32, name="log_h5")
        self.opt = tf.keras.optimizers.legacy.Adam() # tf.keras.optimizers.Adam()

        self._name = name
    
    def call(self, inputs):
        # output = self.c0 + inputs * self.nn.call(inputs)
        return self.nn.call(inputs)
    
    @tf.function
    def ODE(self, t):
        c = self.call(t)
        v = tf.ones_like(t)
        ct = jvp(c, t, v)[0]
        c1, c2, c3, c4, c5, c6, c7, c8 = tf.split(c, 8, axis=-1)
        c1t, c2t, c3t, c4t, c5t, c6t, c7t, c8t = tf.split(ct, 8, axis=-1)
        s1, s2, s3, s4, s5, s6, s7, s8 = tf.split(self.scale, 8, axis=-1)
        
        # coefficients
        k9 = 20
        h9 = 0.2
        XIa = 0.3
        k10 = 0.003
        k10_ = 500
        h10 = 1
        k2 = 2.3
        k2_ = 2000
        k2m = 58
        k2m_ = 210
        h2 = 1.3
        # k8 = 0.00001
        k8 = tf.math.exp(self.log_k8)
        h8 = 0.31
        # ka = 1.2
        ka = tf.math.exp(self.log_ka)
        # k5 = 0.17
        k5 = tf.math.exp(self.log_k5)
        # h5 = 0.31
        h5 = tf.math.exp(self.log_h5)
        k_apc = 0.0014
        # k_apc = tf.math.exp(self.log_k_apc)
        h_apc = 0.1
        k1 = 2.82
        # k1 = tf.math.exp(self.log_k1)
        h11 = 0.2
        k5_10 = 100
        k8_9 = 100
        h5_10 = 100
        h8_9 = 100
        # equations
        Z = k8_9 * s5 * s1 * c5 * c1 / (h8_9 + ka * s7 * c7)
        W = k5_10 * s6 * s2 * c6 * c2 / (h5_10 + ka * s7 * c7)

        # eq1 = c1t - k9 * XIa / s1 + h9 * c1
        # eq2 = c2t - k10 * c1 * s1 / s2 - k10_ * Z / s2 + h10 * c2
        # eq3 = c3t - k2 * c2 * c4 * s2 * s4 / s3 / (c4*s4 + k2m) - k2_ * W * c4 * s4 / s3 / (c4*s4 + k2m_) + h2 * c3
        # eq4 = c4t + k2 * c2 * c4 * s2 / (c4*s4 + k2m) + k2_ * W * c4 / (c4*s4 + k2m_)
        eq5 = c5t - k8 * c3 * s3 / s5 + h8 * c5 + ka * c7 * s7 * (c5 + Z/s5)
        eq6 = c6t - k5 * c3 * s3 / s6 + h5 * c6 + ka * c7 * s7 * (c6 + W/s6)
        # eq7 = c7t - k_apc * c3 * s3 / s7 + h_apc * c7
        # eq8 = c8t - k1 * c3 * s3 / s8
        # return tf.concat([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8], axis=-1)
        return tf.concat([eq5, eq6], axis=-1)


    def train_op(self, t_ode, t_u, u):
        with tf.GradientTape() as tape:
            ODE = self.ODE(t_ode)
            # loss_ode = tf.reduce_mean(ODE ** 2)
            eq5, eq6 = tf.split(ODE, 2, axis=-1)
            # 0.00001, 10, eps = 1e-4
            loss_ode = 0.00001 * tf.reduce_mean(eq5 ** 2) + \
                        100 * tf.reduce_mean(eq6 ** 2)
            u_pred = self.call(t_u)
            loss_u = tf.reduce_mean((u_pred - u) ** 2)
            total_loss = self.eps * loss_ode + loss_u
            eq5_loss = tf.reduce_mean(eq5 ** 2)
            eq6_loss = tf.reduce_mean(eq6 ** 2)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss, loss_ode, loss_u, eq5_loss, eq6_loss
        # return total_loss, loss_ode, loss_u

    def train(self, t_ode, t_u, u, niter=20000):
        t_ode = tf.constant(t_ode, tf.float32)
        t_u = tf.constant(t_u, tf.float32)
        u = tf.constant(u, tf.float32)

        train_op = tf.function(self.train_op)

        loss = []
        # ode_loss = []
        # data_loss = []
        # h5_list = []
        min_loss = 1000
        for i in range(niter):
            loss_value, ode_loss_value, data_loss_value, eq5_loss, eq6_loss = train_op(t_ode, t_u, u)
            # loss_value, ode_loss_value, data_loss_value = train_op(t_ode, t_u, u)
            loss += [loss_value.numpy()]
            # ode_loss += [ode_loss_value.numpy()]
            # data_loss += [data_loss_value.numpy()]
            # h5_list += [tf.math.exp(self.log_h5).numpy()]
            if i % 1000 == 0:
                current_loss = loss[-1]
                print(i, current_loss, ode_loss_value.numpy(), data_loss_value.numpy())
                # print(i, loss[-1], ode_loss[-1], data_loss[-1], eq5_loss.numpy(), eq6_loss.numpy(), tf.math.exp(self.log_h5).numpy())
                # print(i, loss[-1], ode_loss[-1], data_loss[-1])
                if current_loss < min_loss:
                    min_loss = current_loss
                    self.save_weights(
                        filepath="./checkpoints/"+self.name, overwrite=True,
                    )
        return loss # , ode_loss, data_loss, min_loss, h5_list

    def restore(self):
        self.load_weights(filepath="./checkpoints/"+self.name)
