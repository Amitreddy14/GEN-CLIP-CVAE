import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import preprocess as pp
from pycocotools.coco import COCO
import shutil

buzzwords = ["cow", "sheep", "mountain", "hill", "countryside", "grass", "forest", "nature", 
                 "farm", "alpaca", "horse", "landscape", "fence"]