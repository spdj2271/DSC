import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
from tensorflow.keras import layers
from tqdm import tqdm
from utils.utils import *


def model_conv(cfg, load_weights=True):
    input_shape = cfg.INPUT_SHAPE
    hidden_units = cfg.CLUSTER.HIDDEN_UNITS
    weigth_path = cfg.AUTOENCODER.WEIGTH_PATH

    filters = [32, 64, 128, hidden_units]
    pad3 = 'same' if input_shape[0] % 8 == 0 else 'valid'

    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, activation='relu', padding='same', name='layer_conv1')(input)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters[2], kernel_size=3, strides=2, activation='relu', padding=pad3)(x)
    hidden_conv_shape = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed_wol2')(x)

    x = layers.Lambda(lambda x: tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1)), name='embed')(x)
    h = x

    x = layers.Dense(tf.reduce_prod(hidden_conv_shape), activation='relu')(x)
    x = layers.Reshape(hidden_conv_shape)(x)
    x = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, activation='relu', padding=pad3)(x)
    x = layers.Conv2DTranspose(filters[0], kernel_size=5, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(input_shape[2], kernel_size=5, strides=2, padding='same')(x)
    output = layers.Concatenate()([h,
                                   layers.Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    if load_weights:
        model.load_weights(weigth_path)
        print(f'model_conv: weights was loaded, weight path is {weigth_path}')
    return model


def train_base(model, ds_xx, cfg, epoch=None, batchsize=None):
    @tf.function
    def loss_train_base(y_true, y_pred):
        y_true = layers.Flatten()(y_true)
        y_pred = y_pred[:, hidden_units:]
        return losses.mse(y_true, y_pred)

    class pbarCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pbar.update(1)
            pbar.set_postfix(logs)

        def on_train_end(self, logs=None):
            pbar.close()

    # global hidden_units, pbar
    if not epoch:
        epoch = cfg.AUTOENCODER.AUTOENCODER_EPOCHS
    hidden_units = cfg.CLUSTER.HIDDEN_UNITS
    pbar = tqdm(total=epoch)
    weigth_path = cfg.AUTOENCODER.WEIGTH_PATH

    earlystop_callback = EarlyStopping(monitor='loss', min_delta=1e-8, patience=5)
    model.compile(optimizer='adam', loss=loss_train_base)
    model.fit(ds_xx, epochs=epoch, verbose=0, callbacks=[pbarCallback(), earlystop_callback])
    model.save_weights(weigth_path)
