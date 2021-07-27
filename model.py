import tensorflow as tf
from keras.layers import *
import math


# source model
class Sketch2Image(tf.keras.Model):
    def __init__(self):
        super(Sketch2Image, self).__init__()
        self.encoder = tf.keras.Sequential([
            downsample(16, 4, strides=2, apply_batch_normalization=False),
            downsample(16, 4, strides=1, padding='same', apply_batch_normalization=False),
            downsample(32, 4, strides=2),
            downsample(32, 4, strides=1, padding='same'),
            downsample(64, 4, strides=2, apply_batch_normalization=False),
            downsample(64, 4, strides=1, padding='same', apply_batch_normalization=False),
            downsample(128, 4, strides=2),
            downsample(128, 4, strides=1, padding='same'),
            downsample(256, 4, strides=2),
            downsample(256, 4, strides=1, padding='same')
        ])

        self.encoder_out = downsample(512, 4, strides=2)

        self.decoder = tf.keras.Sequential([
            upsample(512, 4, strides=2, apply_dropout=True),
            upsample(256, 4, strides=2, apply_dropout=True),
            upsample(256, 4, strides=1, padding='same', apply_dropout=True),
            upsample(128, 4, strides=2, apply_dropout=True),
            upsample(128, 4, strides=1, padding='same', apply_dropout=True),
            upsample(64, 4, strides=2),
            upsample(64, 4, strides=1, padding='same'),
            upsample(32, 4, strides=2),
            upsample(32, 4, strides=1, padding='same'),
            upsample(16, 4, strides=2),
            upsample(16, 4, strides=1, padding='same')
        ])

        self.c1 = Conv2DTranspose(8, (2, 2), strides=(1, 1), padding='valid')
        self.c2 = Conv2DTranspose(3, (2, 2), strides=(1, 1), padding='valid')

    def call(self, x):
        x = self.encoder(x)
        encoder_out = self.encoder_out(x)
        x = self.decoder(encoder_out)
        x = self.c1(x)
        x = self.c2(x)
        return x


def downsample(filters, size, strides, padding='valid', apply_batch_normalization=True):
    downsample = tf.keras.models.Sequential()
    downsample.add(Conv2D(filters=filters, kernel_size = size, strides=strides,
                          use_bias=False, kernel_initializer='he_normal', padding=padding))
    if apply_batch_normalization:
        downsample.add(BatchNormalization())
    downsample.add(LeakyReLU())
    return downsample


def upsample(filters, size, strides, padding='valid', apply_dropout=False):
    upsample = tf.keras.models.Sequential()
    upsample.add(Conv2DTranspose(filters=filters, kernel_size=size, strides=strides,
                                 use_bias=False, kernel_initializer='he_normal', padding=padding))
    if apply_dropout:
        upsample.add(Dropout(0.1))
    upsample.add(LeakyReLU())
    return upsample

