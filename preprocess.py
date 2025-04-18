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

    print("ids to get: ", ids_to_get)

    # Get image IDs
    image_ids = coco_captions.getImgIds()

    # Load images (get filepaths, and associate with captions)
    images = coco_captions.loadImgs(image_ids)
    filepaths_and_captions = []
    for img in images:
        full_fp = os.path.join(image_directory, img["file_name"])
        annotations = [ann['caption'] for ann in coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=img['id'], iscrowd=None))]
        if check_naturey(annotations, is_small):
            filepaths_and_captions.append((full_fp, annotations))

    # Create a Tensorflow dataset from the filepaths and annotations
    dataset = tf.data.Dataset.from_generator(
        lambda: filepaths_and_captions,
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([None]))
    )

    def load_and_preprocess_image(path, captions):
       image = tf.io.read_file(path)
       image = tf.image.decode_jpeg(image, channels=3)
       image = tf.image.resize(image, [224, 224])
       image = image / 255.0
       return image, captions

    # Create a dataset of ONLY images
    dataset = dataset.map(load_and_preprocess_image)
    train_size = int(len(filepaths_and_captions)*0.75)        