import tensorflow as tf 
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from preprocess import data_loading

## Plot Mask Histogram
## Number of Class  3
'''
img = cv2.imread('masks/M-33-20-D-d-3-3.tif',cv2.IMREAD_COLOR)
img_arr = np.array(img)  
np.unique(img_arr)
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()

cv2.imshow('masks', img)

cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 
'''
##Configuration 
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES=4

TRAIN_PATH = os.path.dirname(os.path.abspath('F:/Personal Projects/Compressed/landcover.ai/images'))

with open('train.txt') as f:
    train_ids = f.readlines()
    
with open('val.txt') as f:
    val_ids = f.readlines()
    
with open('test.txt') as f:
    test_ids = f.readlines()



dataset = data_loading(train_ids, val_ids)
dataset = dataset



STEPS_PER_EPOCH = len(train_ids)//BATCH_SIZE
VALIDATION_STEPS = len(val_ids)//BATCH_SIZE


#Defining U-NET

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, CHANNELS))
s = inputs
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv1 = tf.keras.layers.Dropout(0.1)(conv1)
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
pool1 = tf.keras.layers.MaxPooling2D(2,2)(conv1)

conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
conv2 = tf.keras.layers.Dropout(0.1)(conv2)
conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
pool2 = tf.keras.layers.MaxPooling2D(2,2)(conv2)

conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
conv3 = tf.keras.layers.Dropout(0.1)(conv3)
conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
pool3 = tf.keras.layers.MaxPooling2D(2,2)(conv3)

conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
conv4 = tf.keras.layers.Dropout(0.1)(conv4)
conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
pool4 = tf.keras.layers.MaxPooling2D(2,2)(conv4)

conv5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
conv5 = tf.keras.layers.Dropout(0.1)(conv5)
conv5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

#EXPANSIVE U-NET

u6 = tf.keras.layers.Conv2DTranspose(128,(2,2), strides=(2,2), padding='same')(conv5)
u6 = tf.keras.layers.concatenate([u6, conv4])
conv6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
conv6 = tf.keras.layers.Dropout(0.1)(conv6)
conv6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(conv6)
u7 = tf.keras.layers.concatenate([u7, conv3])
conv7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
conv7 = tf.keras.layers.Dropout(0.1)(conv7)
conv7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)


u8 = tf.keras.layers.Conv2DTranspose(32,(2,2), strides=(2,2), padding='same')(conv7)
u8 = tf.keras.layers.concatenate([u8, conv2])
conv8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
conv8 = tf.keras.layers.Dropout(0.1)(conv8)
conv8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2), strides=(2,2), padding='same')(conv8)
u9 = tf.keras.layers.concatenate([u9, conv1])
conv9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
conv9 = tf.keras.layers.Dropout(0.1)(conv9)
conv9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

outputs = tf.keras.layers.Conv2D(NUM_CLASSES,(1,1), activation='softmax')(conv9)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=10)

checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_model_for_lancover_2.h5',
                                                verbose=1,
                                                save_best_only=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')


results = model.fit(dataset['train'], 
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs = 50, validation_data = dataset['val'],
                    validation_steps = VALIDATION_STEPS,
                    callbacks = [es,checkpoint,tensorboard])


