#import matplotlib.pyplot as plt
#import numpy as np
#import PIL
import tensorflow as tf

import pathlib
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

THIS_DIR = os.path.split(__file__)[0]

data_dir = pathlib.Path(os.path.join(THIS_DIR,'../../Hey-Waldo/64-gray/')).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total images: {image_count}")



# Set parameters for dataset loading
batch_size = 32 
img_height = 64
img_width = 64

# Load the training dataset with the specified options
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)

# Load the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)

# Retrieve the class names
class_names = train_ds.class_names
print(f"Class names: {class_names}")

#AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)






