from glob import glob
import re
from math import ceil

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.utils import to_categorical

from sklearn.utils.class_weight import compute_sample_weight

from tools.image_tools import load_imgs_from_df
from tools.keras_tools import MultiOutputDataGenerator

image_dir = 'data/phenocam_train_images/'
train_sample_size = 20000
random_image_crops = 5
crop_size = 400

validation_fraction = 0.2
target_size = (512,512)
batch_size  = 50

minimum_unique_categories = 30

image_info = pd.read_csv('train_image_annotation/imageant_session2.csv')

# The different targest and their number of classes
output_classes = {'dominant_cover' : 6,
                  'crop_type'      : 7,
                  'crop_status'    : 7}
#output_classes = {'dominant_cover' : 6}

#-------------------------
# some combinations of categories only have a handful of images
# just gonna drop those
class_counts = image_info.groupby(['dominant_cover','crop_type','crop_status']).size().reset_index(name='n')
class_counts['keep'] = class_counts.n.apply(lambda x: True if x>=minimum_unique_categories else False)
class_counts.drop(columns='n', inplace=True)

image_info = image_info.merge(class_counts, how='left', on=['dominant_cover','crop_type','crop_status'])
image_info = image_info[image_info.keep]

#-------------------------
# Setup validation split
total_validation_images = int(len(image_info) * validation_fraction)

# First put all images from held out sites into the validation set
image_info['validation_site'] = image_info.file.apply(lambda f: bool(re.search(r'(arsmorris2)|(mandani2)|(cafboydnorthltar01)', f)))
validation_images = image_info[image_info.validation_site]

# Add in a random sample of remaining images to get to the total validation fraction
# images from validation sites excluded here by setting weight to 0
image_info['validation_weight'] = image_info.validation_site.apply(lambda val_site: 0 if val_site else 1)
validation_images = validation_images.append(image_info.sample(n= total_validation_images - len(validation_images), replace=False, weights='validation_weight')) 

# Training images are ones that are left
train_images = image_info[~image_info.index.isin(validation_images.index)]

# assure no validtion sites in the training data, and all images in each set are unique
assert train_images.validation_site.sum() == 0, 'validation sites in training dataframe'
assert train_images.index.nunique() == len(train_images), 'duplicates in training dataframe'
assert validation_images.index.nunique() == len(validation_images), 'duplicates in validation dataframe'

#-------------------------
# expand training by random sampling, weighted so that low sample size category images are repeated.
# This makes it so sample sizes are even in training

# Use the unique combinations across the 3 categories. 
train_images['unique_category'] = train_images.dominant_cover.astype(str) + train_images.crop_type.astype(str) + train_images.crop_status.astype(str)
train_images['sample_weight'] = compute_sample_weight('balanced', train_images.unique_category)
train_images = train_images.sample(n=train_sample_size, replace=True, weights='sample_weight')

#-------------------------
# Generate numpy arrays of all images. Each image is repeated several times from the sampling
train_x = load_imgs_from_df(train_images, x_col='file', img_dir=image_dir, 
                            target_size=target_size, data_format='channels_last')

# Multiple output model means a dictionary for the targets
# these names match the final model layer names
train_y = {c:to_categorical(train_images[c]) for c in output_classes.keys()}


train_generator = MultiOutputDataGenerator(preprocessing_function=None, # scaling done via Rescale layer
                                     vertical_flip = True,
                                     horizontal_flip = True,
                                     rotation_range = 45,
                                     zoom_range = 0.25,
                                     width_shift_range = [-0.25,0,0.25],
                                     height_shift_range = [-0.25,0,0.25],
                                     #shear_range = 45,
                                     brightness_range = [0.5,1.5],
                                     fill_mode='reflect').flow(
                                         x = train_x,
                                         y = train_y,
                                         shuffle = True,
                                         batch_size = batch_size,
                                         )

# small sample for testing                                         
#validation_images = validation_images.sample(500)                      
val_x =  load_imgs_from_df(validation_images, x_col='file', img_dir=image_dir, 
                            target_size=target_size, data_format='channels_last')
val_y = {c:to_categorical(validation_images[c]) for c in output_classes.keys()}

#-------------------------
# The keras model
#-------------------------
# basiccnn from https://towardsdatascience.com/different-ways-of-improving-training-accuracy-c526db15a5b2
base_model = keras.Sequential()
base_model.add(Input(shape=target_size + (3,)))
base_model.add(Rescaling(scale = 1./127.5))
base_model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv1a'))
base_model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv1b'))
base_model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same', name='max_pool1'))

base_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv2a'))
base_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv2b'))
base_model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same', name='max_pool2'))

base_model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv3a'))
base_model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                        strides=(2, 2), activation='relu', name='conv3b'))
base_model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same', name='max_pool3'))

base_model.add(keras.layers.Dense(512, activation='relu', name='main_dense1'))
base_model.add(keras.layers.Dropout(0.5, name='main_dropout1'))
base_model.add(keras.layers.Dense(512, activation='relu', name='main_dense2'))
base_model.add(keras.layers.Flatten(name='main_flatten'))

def build_category_model(prior_step, class_n, name):
    x = keras.layers.Dense(256, activation = 'relu', name=name+'_dense1')(prior_step)
    x = keras.layers.Dropout(0.5, name=name+'_dropout1')(x)
    x = keras.layers.Dense(256, activation = 'relu', name=name+'_dense2')(x)
    x = keras.layers.Dropout(0.5, name=name+'_dropout2')(x)
    x = keras.layers.Dense(class_n,  activation = 'softmax', name=name)(x)
    return(x)

sub_models = [build_category_model(base_model.output, class_n, name) for name, class_n in output_classes.items()]

full_model = keras.Model(base_model.input, sub_models)

full_model.compile(optimizer = keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])
print(full_model.summary())

full_model.fit(train_generator,
          validation_data= (val_x,val_y),
          #class_weight = weights,
          steps_per_epoch=ceil(train_sample_size/batch_size), # this is not automatic cause of custom generator
          epochs=30,
          use_multiprocessing=False)
