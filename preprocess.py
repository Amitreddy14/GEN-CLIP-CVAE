import os
import tensorflow as tf
import clip_wrapper as cw
import glob
from pycocotools.coco import COCO

small_buzzwords = ["cow", "sheep"]
large_buzzwords = ["cow", "sheep", "mountain", "hill", "countryside", "grass", "forest", "nature", 
                 "farm", "alpacca", "horse", "landscape", "fence"]