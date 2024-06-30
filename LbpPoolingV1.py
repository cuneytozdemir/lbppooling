# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 00:27:51 2022

@author: LENOVO
"""

import numpy as np
import tensorflow as tf
import math
import os
import random
import itertools
from tensorflow.python.ops import gen_nn_ops
import numba as nb
import matplotlib.pyplot as plt

random.seed(91)

mask = [[64, 128, 1], [32, 0, 2], [16, 8, 4]]
masked = np.array(mask).ravel()

@nb.njit()
def lbppool(a):
    u, g = a.shape
    if not (u == 3 and g == 3):
        f = np.zeros((5, 5))
        f[:u, :g] = f[:u, :g]+a
        f = f[:3, :3]
        f = f.ravel()
        s = (np.where(f >= f[4], 1, 0)*masked).sum()
        return (s* f.std() + (1 - s)*f.max())/255.0
    else:
        a = a.ravel()
        s = (np.where(a >= a[4], 1, 0)*masked).sum()
        return (s* a.std() + (1 - s)*a.max())/255.0


    
    
class LbpPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(3, 3), strides=(2, 2), padding='SAME', data_format='channels_last', **kwargs):
        super(LbpPooling2D, self).__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'


    def build(self, input_shape):
        super(LbpPooling2D, self).build(input_shape)

    def _pooling_function(self, x, name=None):
        input_shape = tf.keras.backend.int_shape(x)
        b, r, c, channel = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        pr, pc = self.pool_size
        sr, sc = self.strides
        # compute number of windows
        num_r = math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc

        
        def _mid_pool(inputs,istrain):
            input_shape = inputs.shape
            batch = input_shape[0]
            # reshape
            w = np.transpose(inputs, (0, 3, 1, 2))
            w = np.reshape(w, (batch*channel, r, c))
            re = np.zeros((batch*channel, num_r, num_c), dtype=np.float32)
            
        
            for i, j in itertools.product(range(num_r), range(num_c)):
                re[:, i, j] = np.array(tuple(map(lambda x: lbppool(x), w[:, i * sr:i*sr+pr, j*sc:j*sc+pc])))


            re = np.reshape(re, (batch, channel, num_r, num_c))
            re = np.transpose(re, (0, 2, 3, 1))
            return re

        def custom_grad(op, grad):
            if self.data_format == 'NHWC':
                ksizes = [1, self.pool_size[0], self.pool_size[1], 1]
                strides = [1, self.strides[0], self.strides[1], 1]
            else:
                ksizes = [1, 1, self.pool_size[0], self.pool_size[1]]
                strides = [1, 1, self.strides[0], self.strides[1]]
            return gen_nn_ops.max_pool_grad_v2(
                op.inputs[0],
                op.outputs[0],
                grad,
                ksizes,
                strides,
                self.padding,
                data_format=self.data_format
            ), tf.constant(0.0)

        def py_func(func, inp, Tout, stateful=True, name=None, grad=None, rnd_name=None):
            # Need to generate a unique name to avoid duplicates:
            tf.compat.v1.RegisterGradient(rnd_name)(grad)
            g = tf.compat.v1.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.compat.v1.py_func(func, inp, Tout, stateful=stateful, name=name)

        def _mid_range_pool(x, name=None):
            rnd_name = 'LbpPooling2D' + str(np.random.randint(0, 1E+8))

            with tf.compat.v1.name_scope(name, "mod", [x]) as name:
                z = py_func(_mid_pool,
                            [x,1],
                            [tf.float32],
                            name=name,
                            grad=custom_grad, rnd_name=rnd_name)[0]
                z.set_shape((b, num_r, num_c, channel))
                return z

        return _mid_range_pool(x, name)

    def compute_output_shape(self, input_shape):
        r, c = input_shape[1], input_shape[2]
        sr, sc = self.strides
        num_r = math.ceil(r/sr) if self.padding == 'SAME' else r//sr
        num_c = math.ceil(c/sc) if self.padding == 'SAME' else c//sc
        return (input_shape[0], num_r, num_c, input_shape[3])

    def call(self, inputs):
        # K.in_train_phase(self._tf_pooling_function(inputs), self._pooling_function_test(inputs))
        output = self._pooling_function(inputs)
        return output

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'strides': self.strides
        }
        base_config = super(LbpPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_model(pooling):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3,activation="relu",input_shape=input_shape))
    # model.add(tf.keras.layers.SpatialDropout2D(0.15))
    model.add(pooling)
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3,activation='relu'))
    # model.add(tf.keras.layers.SpatialDropout2D(0.1))
    model.add(pooling)
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.GlobalAveragePooling2D())
    # model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    return model


batch_size = 32
epochs = 3

num_classes = 10
channel=3
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)
input_shape = (img_rows, img_cols, channel)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min')
callbacks_list = [reduceLROnPlat]

print('custom pooling')
modelcus = get_model(LbpPooling2D())
modelcustom = modelcus.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        # validation_data=(x_test, y_test)
                        validation_split=0.1
                        ,callbacks=callbacks_list)

ctest_loss, ctest_acc = modelcus.evaluate(x_test, y_test, verbose=1)
# ctrain_loss, ctrain_acc = model.evaluate(x_train, y_train, verbose=1)

print('max pooling')
modelm = get_model(tf.keras.layers.MaxPooling2D())
modelmax = modelm.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     # validation_data=(x_test, y_test)
                      validation_split=0.1
                     ,callbacks=callbacks_list)

mtest_loss, mtest_acc = modelm.evaluate(x_test, y_test, verbose=1)
# mtrain_loss, mtrain_acc = model.evaluate(x_train, y_train, verbose=1)


print('avg pooling')
modela = get_model(tf.keras.layers.AveragePooling2D())
modelavg = modela.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     # validation_data=(x_test, y_test)
                      validation_split=0.1
                    ,callbacks=callbacks_list
                     )

atest_loss, atest_acc = modela.evaluate(x_test, y_test, verbose=1)
# atrain_loss, atrain_acc = model.evaluate(x_train, y_train, verbose=1)


print("LBP accuracy(test): {:.4}".format(ctest_acc))
print("Max accuracy(test): {:.4}".format(mtest_acc))
print("Avg accuracy(test): {:.4}".format(atest_acc))
print()

