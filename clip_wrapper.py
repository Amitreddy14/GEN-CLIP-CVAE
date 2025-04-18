import torch
import clip
from PIL import Image
import tensorflow as tf

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)