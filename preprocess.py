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

def load_coco_data(image_directory, captions_file, is_small: bool):
    # Initialize COCO with annotations
    coco_captions = COCO(captions_file)

    full_paths = glob.glob(os.path.join(image_directory, "*.jpg"))
    names = [p.split("/")[-1].replace(".jpg", "") for p in full_paths]
    ids_to_get = [int(n) for n in names]

    