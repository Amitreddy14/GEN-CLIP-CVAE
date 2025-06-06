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

    # Define Python Function to get image embeddings (this will return a numpy array)
    def get_clip_im_embeddings(images):
        # If single image
        if len(images.shape) == 3:
            return cw.batch_get_image_encodings(tf.expand_dims(images, axis=0)) 
        else:
            return cw.batch_get_image_encodings(images)

    # py_function to use the tensors in the dataset to get the embeddings
    def tf_py_function_clip_im_embeddings(images, captions):
        clip_im_embeddings = tf.py_function(get_clip_im_embeddings, [images], tf.float32)
        clip_im_embeddings.set_shape((1, 512))
        return images, clip_im_embeddings, captions

    dataset = dataset.map(tf_py_function_clip_im_embeddings)

    def get_clip_text_embeddings(captions):
        # If single image
        token_list = []
        for i in range(captions.shape[0]):
            token_list += [captions[i].cpu().numpy().decode()]
        return cw.get_text_encoding(token_list)

    def tf_py_function_clip_text_embeddings(images, clip_im_embeds, captions):
        clip_txt_embeddings = tf.py_function(get_clip_text_embeddings, [captions], tf.float32)
        clip_txt_embeddings.set_shape((5, 512))
        return images, clip_im_embeds, captions, clip_txt_embeddings
    
    dataset = dataset.map(tf_py_function_clip_text_embeddings)      

    def get_tokens(captions):
        token_list = []
        for i in range(captions.shape[0]):
            token_list += [captions[i].cpu().numpy().decode()]
        return cw.get_tokens(token_list)

    def tf_py_function_tokens(images, clip_im_embeds, captions, clip_txt_embeds):
        tokens = tf.py_function(get_tokens, [captions], tf.float32)
        return images, clip_im_embeds, captions, clip_txt_embeds, tokens   

    dataset = dataset.map(tf_py_function_tokens)

    train_dataset = dataset.take(train_size)
    valid_dataset = dataset.skip(train_size)

    print("\nSuccessfully initialized!")
    return train_dataset, valid_dataset

def get_64x64_images(dataset):
    def resize(image, clip_im_embeds, captions, clip_txt_embeds, tokens):
        return tf.image.resize(image, [64, 64])
    return dataset.map(resize) 

def get_64x64_images_and_embeddings(dataset):
    def resize(image, clip_im_embed, captions, clip_txt_embeds, tokens):
        return tf.image.resize(image, [64, 64]), clip_im_embed
    return dataset.map(resize)

def get_128x128_images_and_embeddings(dataset):
    def resize(image, clip_im_embed, captions, clip_txt_embeds, tokens):
        return tf.image.resize(image, [128, 128]), clip_im_embed
    return dataset.map(resize)

def get_64x64_images_and_5_text_embeddings(dataset):
    def create_miniset(image, clip_im_embeds, captions, clip_txt_embeds, tokens):
        miniset = tf.data.Dataset.from_tensor_slices(clip_txt_embeds)
        def populate(clip_txt_embed):
            return tf.image.resize(image, [64, 64]), tf.expand_dims(clip_txt_embed, axis=0)
        return miniset.map(populate)
    return dataset.flat_map(create_miniset)

def get_64x64_images_and_1_text_embedding(dataset):
    def populate(image, clip_im_embeds, captions, clip_txt_embeds, tokens):
        return tf.image.resize(image, [64, 64]), clip_txt_embeds[0]
    return dataset.map(populate)

def get_128x128_images_and_5_text_embeddings(dataset):
    def create_miniset(image, clip_im_embeds, captions, clip_txt_embeds, tokens):
        miniset = tf.data.Dataset.from_tensor_slices(clip_txt_embeds)
        def populate(clip_txt_embed):
            return tf.image.resize(image, [128, 128]), tf.expand_dims(clip_txt_embed, axis=0)
        return miniset.map(populate)
    return dataset.flat_map(create_miniset)

def get_64x64_images_and_1_text_embedding(dataset):
    def populate(image, clip_im_embeds, captions, clip_txt_embeds, tokens):
        return tf.image.resize(image, [128, 128]), clip_txt_embeds[0]
    return dataset.map(populate)  