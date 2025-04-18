import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
import clip_wrapper

class ClipCVAE(tf.keras.Model):
  def __init__(self, input_shape,  latent_dim, dropout_rate):
    super(ClipCVAE, self).__init__()
    self.latent_dim = latent_dim
    self.shape_input = input_shape
    self.dropout_prob = dropout_rate