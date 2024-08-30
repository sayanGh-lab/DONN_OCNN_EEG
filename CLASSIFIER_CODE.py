# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:20:17 2024

@author: SAYAN GHOSH
"""

import os, math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io

data5=scipy.io.loadmat('D:/Lab_work_24/aug/paper-review/BONN/D_E/model_dataset.mat')

#data6=scipy.io.loadmat('D:/lab_work-2023/september(mount sinai)/deep_oscillator/omega.mat')

EEG_data1=data5["main_EEG"] # 4 channels EEG TRAINING Data
train_label=data5["label"] # corresponding target o/p

print(EEG_data1.shape)
print(train_label.shape)

EEG_data = np.expand_dims(EEG_data1, axis=-1)


print(EEG_data.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(EEG_data[0:400,], train_label[0:400,]/2.8, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

@tf.function
def real_cal(r, phi):
    return r * tf.math.cos(phi)

@tf.function
def imag_cal(r, phi):
    return r * tf.math.sin(phi)

@tf.function
def oscillator_loop(X_r, X_i, omegas, num_steps):
    # batch_size x timesteps X dim
    r_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True) # creates empty array to save r_t
    phi_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    r_t = tf.ones((X_r.shape[0], X_r.shape[-1])) # Initializing r_t
    phis = tf.zeros((X_r.shape[0], X_r.shape[-1])) # Initislizing phi_t
    dt = 1/173.61
    input_scaler = 0.1
    beta=0.01

    for t in tf.range(num_steps):
        input_r = input_scaler*X_r[:,t,:]*tf.math.cos(phis)
        input_phi = input_scaler*X_i[:,t,:]*tf.math.sin(phis)
        r_t = r_t + ((1 - beta*tf.square(r_t)) * r_t + input_r) * dt
        phis = phis + (omegas - input_phi) * dt
        r_arr = r_arr.write(r_arr.size(), r_t)  #1000,1,2
        phi_arr = phi_arr.write(phi_arr.size(), phis)
    r_arr = tf.transpose(r_arr.stack(), [1, 0, 2])  # Changing dimensions to 1,1000,2
    phi_arr = tf.transpose(phi_arr.stack(), [1, 0, 2])
    return r_arr, phi_arr

class Hopf(tf.keras.layers.Layer):

    def __init__(self, units, num_steps, min_omega=0.1,
                 max_omega=70.1, **kwargs):
        super(Hopf, self).__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.omegas = tf.linspace(min_omega, max_omega, self.units) * (2*3.1415)
        self.omegas = tf.cast(tf.expand_dims(self.omegas, 0), 'float32')


    def call(self, X_r, X_i):
        r, phi = oscillator_loop(X_r, X_i, self.omegas, self.num_steps)
        z_real = real_cal(r, phi)
        z_imag = imag_cal(r, phi)
        return z_real, z_imag

duration = 1000

class Model(tf.keras.Model):

    def __init__(self, units1, units2,units3,units4,units5, **kwargs):

        super(Model, self).__init__(**kwargs)

        self.d1_r = tf.keras.layers.Dense(units1,activation='relu') # 1st relu layer(real) with neuron=units1
        self.d1_i = tf.keras.layers.Dense(units1,activation='relu')# 1st relu layer(imag) with neuron=units1

        self.osc1 = Hopf(units2, num_steps=duration,min_omega=2.0, max_omega=35) # 1st osc layer=units2=units1

        #self.d_r = tf.keras.layers.Dense(units3,activation='relu') # 1st relu layer(real) with neuron=units1
        self.d_i = tf.keras.layers.Dense(units3,activation='tanh')# 1st relu layer(imag) with neuron=units1


        #self.d2_r = tf.keras.layers.Dense(units4, activation='linear') #last tanh layer,units5
        #self.d2_i = tf.keras.layers.Dense(units4, activation='linear') #last tanh layer,units5


        self.out_dense = tf.keras.layers.Dense(units5, activation='linear')# output node, units6, with tanh

    def call(self, X):

        out1_r = tf.keras.layers.TimeDistributed(self.d1_r)(X)
        out1_i = tf.keras.layers.TimeDistributed(self.d1_i)(X)

        z1_r, z1_i = self.osc1(out1_r, out1_i)


        concat_inp=tf.concat([z1_r, z1_i],2)
        #out2_r = tf.keras.layers.TimeDistributed(self.d_r)(z1_r)
        out2_i = tf.keras.layers.TimeDistributed(self.d_i)(concat_inp)

        #concat_inp=tf.concat([out2_r,out2_i],2)


        #out3_r = tf.keras.layers.TimeDistributed(self.d2_r)(out2_r)
        #out3_i = tf.keras.layers.TimeDistributed(self.d2_i)(out2_i)
        #out3=tf.concat([out3_r,out3_i],2)

        out_final = tf.keras.layers.TimeDistributed(self.out_dense)(out2_i)
        return out_final

model = Model(80,80,40,30,2)
optimizer = tf.keras.optimizers.Adam(0.01)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath='D:/Lab_work_24/aug/paper-review/BONN/D_E/',
                                    save_weights_only=True,
                                    save_best_only = True)


model.compile(optimizer, 'mse')



history=model.fit(X_train,y_train,epochs=50, batch_size = 32, validation_split=0.2,
          callbacks=[model_checkpoint_callback])

out = model.predict(X_train, batch_size=1)


plt.plot(out[45], label='pre')
plt.plot(y_train[45], label ='desire')
plt.legend()
plt.show()



# plt.plot(out[60,:])
# plt.show()

c=0

for i in range(320):
    a=sum(out[i,:])
    indices = np.where(a == a.max())
    print(indices)
    b=sum(y_train[i,:])
    indices1 = np.where(b == b.max())
    print(indices1)

    if indices==indices1:
       c=c+1

print(c)


model.load_weights('D:/Lab_work_24/aug/paper-review/BONN/D_E/')


out_test = model.predict(X_test, batch_size=1)

c_test=0

for ii in range(80):
    a_test=sum(out_test[ii,:])
    print(a_test)
    indices_test = np.where(a_test == a_test.max())
    print(indices_test)
    b_test=sum(y_test[ii,:])
    indices1_test = np.where(b_test == b_test.max())
    print(indices1_test)

    if indices_test==indices1_test:
       c_test=c_test+1

print(c_test)