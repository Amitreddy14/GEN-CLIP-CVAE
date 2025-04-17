import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
import clip_wrapper

# 64x64 Model for Reference
# Please check the Final Notebooks for the most up-to-date, accurate model
BATCH_SIZE = 16
class ClipCVAE(tf.keras.Model):