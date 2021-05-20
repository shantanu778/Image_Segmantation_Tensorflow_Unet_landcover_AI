import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
#from unet_train import display_sample


BASE_DIR = os.path.dirname(os.path.abspath('F:/Personal Projects/Compressed/landcover.ai/images'))

model = tf.keras.models.load_model('unet_model_for_lancover_2.h5')
model.summary()

IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

with open('test.txt') as f:
    test_ids = f.readlines()
    
    
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
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    
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



def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def show_predictions(dataset=None, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    Accuracy =[]
    if dataset:
        for image, mask in dataset:
            _ , acc = model.evaluate(image, mask)
            Accuracy.append(acc)
            pred_mask = model.predict(image)
            y_pred_argmax = np.argmax(pred_mask, axis = 3)
            
            
            #display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
        inference = model.predict(one_img_batch)
        # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
        pred_mask = create_mask(inference)
        # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
        display_sample([sample_image[0], sample_mask[0],
                        pred_mask[0]])
        
    return np.mean(Accuracy)


test_paths = [BASE_DIR+'\\output\\'+i.rstrip()+'.jpg' for i in  test_ids]
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = test_dataset.map(parse_image)

dataset = {'test': test_dataset}
    
dataset['test'] = dataset['test'].map(load_image_test)
dataset['test'] = dataset['test'].repeat()
dataset['test'] = dataset['test'].batch(BATCH_SIZE)
dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)


    

#display_sample = display_sample([sample_image[1], sample_mask[1]])
y_pred = model.predict(dataset['test'], steps=len(test_ids)//BATCH_SIZE)
y_pred = np.argmax(y_pred, axis =3)

_ , acc = model.evaluate(dataset['test'], steps=len(test_ids)//BATCH_SIZE)
print("Accuracy: ", (acc*100.0),"%")

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap = 'jet')
        plt.axis('off')
    
    return plt.show()

for image, mask in dataset['test'].take(2):
    sample_image, sample_mask = image, mask
    

display_sample([sample_image[20], sample_mask[20], np.expand_dims(y_pred[52], axis = 2)])

'''
acc = show_predictions(dataset['test'])
print(acc)


"""

model.load_weights('unet_model_for_lancover.h5')


test_path = f'{TRAIN_PATH}\images\{TEST_IDS}'

X_test = cv2.imread(test_path)[:,:,:CHANNELS]

X_test = cv2.resize(X_test, (IMG_WIDTH, IMG_HEIGHT))

X_test = X_test/255.0

X_test = np.expand_dims(X_test, axis =0)

predict = model.predict(X_test)

y_pred_argmax = np.argmax(predict, axis = 3)


y_test_path = f'{TRAIN_PATH}\masks\{TEST_IDS}'

y_test = cv2.imread(y_test_path,0) 

y_test = cv2.resize(y_test, (IMG_WIDTH, IMG_HEIGHT))

y_test = np.expand_dims(y_test, axis =0)

y_test_cat = tf.keras.utils.to_categorical(
    y_test, num_classes=4, dtype='float32'
)

_ , acc = model.evaluate(X_test, y_test_cat)
print("Accuracy: ", (acc*100.0),"%")

plt.imshow(X_test[0, :,:,0], cmap='gray')
plt.imshow(y_test.reshape(IMG_WIDTH,IMG_HEIGHT), cmap='gray')


import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
predict = model.predict(X_test)
y_pred_argmax = np.argmax(predict, axis = 3)[0,:,:]


#Ploting
plt.figure(figsize=(12,8))
plt.subplot(231)
plt.imshow(test_img[:,:,0], cmap='gray')
plt.title("Testing Image")
plt.figure(figsize=(12,8))
plt.subplot(232)
plt.imshow(ground_truth, cmap='jet')
plt.title("Ground Truth")
plt.figure(figsize=(12,8))
plt.subplot(233)
plt.imshow(y_pred_argmax, cmap='jet')
plt.title("Predicted Image")

"""
'''