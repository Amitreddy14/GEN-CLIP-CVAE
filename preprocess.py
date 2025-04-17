import os
import tensorflow as tf
import clip_wrapper as cw
import glob
from pycocotools.coco import COCO

small_buzzwords = ["cow", "sheep"]
large_buzzwords = ["cow", "sheep", "mountain", "hill", "countryside", "grass", "forest", "nature", 
                 "farm", "alpacca", "horse", "landscape", "fence"]

def check_naturey(captions: list[str], is_small: bool):
    """Returns true if a caption for this image has one of our buzz words, false otherwise"""
    buzzwords = small_buzzwords if is_small else large_buzzwords

    def match(word):
        for capt in captions:
            if word in capt.lower():
                return True
        return False
    truth_map = [match(word) for word in buzzwords]
    return any(truth_map)