import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import preprocess as pp
from pycocotools.coco import COCO
import shutil

buzzwords = ["cow", "sheep", "mountain", "hill", "countryside", "grass", "forest", "nature", 
                 "farm", "alpaca", "horse", "landscape", "fence"]

def check_naturey(captions: list[str]):
    #returns true if a caption for this image has one of our buzz words, false otherwise
    def match(word):
        for capt in captions:
            if word in capt.lower():
                return True