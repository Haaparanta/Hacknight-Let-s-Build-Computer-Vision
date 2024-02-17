from tensorflow import keras
from tensorflow.keras import layers
import PIL
import pathlib


import os
import tensorflow as tf

THIS_DIR = os.path.split(__file__)[0]
img_height, img_width = 256, 256  # Adjust these values based on your actual image dimensions

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       1)),  # Note the '1' here for grayscale
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Rescaling(1./255),
    ]
)

def load_and_augment_image(image_path, save_path, data_augmentation, num_augmented_images=5):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.expand_dims(img, 0)

    for i in range(num_augmented_images):
        augmented_img = data_augmentation(img)
        augmented_img = tf.squeeze(augmented_img, 0) 
        save_file_path = os.path.join(save_path, f"augmented_{i}_{os.path.basename(image_path)}")
        tf.keras.utils.save_img(save_file_path, augmented_img)

source_folder = os.path.join(THIS_DIR,'../../Hey-Waldo/256-gray/waldo')
output_folder = os.path.join(THIS_DIR,'../../Hey-Waldo/256-gray/waldo')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all images in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        load_and_augment_image(image_path, output_folder, data_augmentation)
