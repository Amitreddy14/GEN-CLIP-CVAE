import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
import clip_wrapper

# 64x64 Model for Reference
# Please check the Final Notebooks for the most up-to-date, accurate model
BATCH_SIZE = 16
class ClipCVAE(tf.keras.Model):

    def __init__(self, input_shape,  latent_dim, dropout_rate):
    super(ClipCVAE, self).__init__()
    self.latent_dim = latent_dim
    self.shape_input = input_shape
    self.dropout_prob = dropout_rate

    # Dense layer used to shrink the embedding before inputting it in the decoder
    self.embedding_shrinker = tf.keras.layers.Dense(32)

    #First part of the encoder before embedding concatenation
    self.encoder_part1= tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=self.shape_input, name="encoder_part1_inputlayer"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(256,5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d3"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal(), name="conv2d4"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten()])
    
    #Second part of the encoder after concatenation
    self.encoder_part2 = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, kernel_initializer=tf.keras.initializers.HeNormal(), activation = "leaky_relu"),
            tf.keras.layers.Dense(2048, kernel_initializer=tf.keras.initializers.HeNormal(), activation="leaky_relu"),
            tf.keras.layers.Dense(1024, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()

        ]
    )

    self.mu = tf.keras.layers.Dense(self.latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())
    self.logv = tf.keras.layers.Dense(self.latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8 * 8 * 256, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((8,8,256)),
            tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(256, 5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(32, 5, strides=1, padding="same", kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(3, 5, strides=1, padding="same", activation="sigmoid", kernel_initializer=tf.keras.initializers.HeNormal())
        ]
    )