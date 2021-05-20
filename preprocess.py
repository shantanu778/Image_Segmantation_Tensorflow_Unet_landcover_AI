import os
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath('F:/Personal Projects/Compressed/landcover.ai/images'))
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32

# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 500


    
    
    
def parse_image(train_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    #img_path = os.path.join(BASE_DIR, 'output')
    image = tf.io.read_file(train_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    #print(image)

    # For one Image path:
    # ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # ADE_train_00000001_m.png
    
    mask_path = tf.strings.regex_replace(train_path, ".jpg", "_m.png")
    #print(mask_path)
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}




def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_WIDTH, IMG_HEIGHT))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_WIDTH, IMG_HEIGHT))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask



def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_WIDTH, IMG_HEIGHT))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_WIDTH, IMG_HEIGHT))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def data_loading(train_ids, val_ids):
    train_paths = [BASE_DIR+'\\output\\'+i.rstrip()+'.jpg' for i in  train_ids]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(parse_image)
    #train_dataset = list(map(parse_image, train_ids))
    
    #val_dataset = list(map(parse_image, val_ids))
    val_paths = [BASE_DIR+'\\output\\'+i.rstrip()+'.jpg' for i in  val_ids]
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
    val_dataset = val_dataset.map(parse_image)
    
    dataset = {"train": train_dataset, "val": val_dataset}
    
    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
    
    #-- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
    
    return dataset
    
    
if __name__ == '__main__':
    data_loading()
