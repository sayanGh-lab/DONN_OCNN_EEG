import tensorflow as tf
import numpy as np
import cv2

def real_cal(r, phi):
    return r * tf.math.cos(phi)

def oscillator_loop(X_r, X_i, omegas, num_steps):
    #X_r - (bs, T, d, d, np, nk)
    r_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    phi_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    (bs, T, d, d, nk) = X_r.shape
    r_t = tf.ones((bs, d, d, nk))
    phis = tf.zeros((bs, d, d, nk))
    dt = 0.01
    input_scaler = 20
    beta1 = 1
    for t in tf.range(num_steps):
        input_r = input_scaler*X_r[:,t,:,:,:]*tf.math.cos(phis)
        input_phi = input_scaler*X_i[:,t,:,:,:]*tf.math.sin(phis)
        r_t = r_t + ((1 - beta1*tf.square(r_t)) * r_t + input_r) * dt
        phis = phis + (omegas - input_phi) * dt
        r_arr = r_arr.write(r_arr.size(), r_t)
        phi_arr = phi_arr.write(phi_arr.size(), phis)
    r_arr = tf.transpose(r_arr.stack(), [1, 0, 2, 3, 4])
    phi_arr = tf.transpose(phi_arr.stack(), [1, 0, 2, 3, 4])
    return r_arr, phi_arr

class Hopf(tf.keras.layers.Layer):

    def __init__(self, dim, depth, num_steps, min_omega=0.1,
                 max_omega=1, **kwargs):

        super(Hopf, self).__init__(**kwargs)

        self.dim = dim
        self.num_filters = depth
        self.num_steps = num_steps

        #omegas = tf.random.uniform((1, (dim**2)*depth), min_omega, max_omega) * (2*3.1415)
        omegas = np.random.uniform(min_omega, max_omega, (1, (dim**2)*depth)) * (2*3.1415)
        omegas = cv2.GaussianBlur(omegas.reshape(dim, dim, depth), (3,3), 1)
        #omegas = tf.linspace(min_omega, max_omega, (dim**2)*depth) * (2*3.1415)        
        self.omega_param = tf.Variable(omegas, trainable=False, dtype='float32')
        self.omega_param = tf.reshape(self.omega_param, (dim, dim, depth))

    def call(self, X_r, X_i):
        r, phi = oscillator_loop(X_r, X_i, self.omega_param, self.num_steps)
        return r * tf.math.cos(phi), r * tf.math.sin(phi)