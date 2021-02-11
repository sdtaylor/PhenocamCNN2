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
train_sample_size = 100
random_image_crops = 5
crop_size = 400

validation_fraction = 0.2
target_size = (32,32)
batch_size  = 10

image_info = pd.read_csv('train_image_annotation/imageant_session2.csv')


########################################
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

# expand training by random sampling, weighted so that low sample size category images are repeated.
# This makes it so sample sizes are even in training

# Use the unique combinations across the 3 categories. 
train_images['unique_category'] = train_images.dominant_cover.astype(str) + train_images.crop_type.astype(str) + train_images.crop_status.astype(str)
train_images['sample_weight'] = compute_sample_weight('balanced', train_images.unique_category)
train_images = train_images.sample(n=train_sample_size, replace=True, weights='sample_weight')


# Generate numpy arrays of all images. Each image is repeated several times from the sampling
train_x = load_imgs_from_df(train_images, x_col='file', img_dir=image_dir, 
                            target_size=target_size, data_format='channels_last')

# Multiple output model means a dictionary for the targets
# these names match the final model layer names
train_y ={c:to_categorical(train_images[c]) for c in ['dominant_cover','crop_type','crop_status']}


train_generator = MultiOutputDataGenerator(preprocessing_function=None, # scaling done via Rescale layer
                                     vertical_flip = True,
                                     horizontal_flip = True,
                                     rotation_range = 45,
                                     #zoom_range = 0.25,
                                     width_shift_range = [-0.25,0,0.25],
                                     height_shift_range = [-0.25,0,0.25],
                                     shear_range = 45,
                                     #brightness_range = [0.2,1],
                                     fill_mode='reflect').flow(
                                         x = train_x,
                                         y = train_y,
                                         shuffle = True,
                                         batch_size = batch_size,
                                         )

# No random transformations for test images                                        
#val_x = scale_images(val_x)
#validation_generator  = ImageDataGenerator(preprocessing_function=scale_images).flow(
#                                         x = val_x,
#                                         y = val_y,
#                                         shuffle = False,
#                                         batch_size = batch_size,
#                                         )

# Example from https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg
#base_model = VGG16_Places365(


#input_layer =  Rescaling(scale = 1./255, input_shape=target_size + (3,))
#input_layer = Input(tensor=input_layer)

base_model = keras.applications.VGG16(
    weights=None,  # Load weights pre-trained on ImageNet.
    input_shape= target_size + (3,),
    #input_tensor = input_layer,
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

def build_category_model(prior_step, class_n, name):
    x = keras.layers.GlobalMaxPooling2D()(prior_step)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(class_n,  activation = 'softmax', name=name)(x)
    return(x)


dominant_cover_model = build_category_model(base_model.output, 6, 'dominant_cover')
crop_type_model      = build_category_model(base_model.output, 7, 'crop_type')
crop_status_model    = build_category_model(base_model.output, 7, 'crop_status')

full_model = keras.Model(base_model.input, [dominant_cover_model,crop_type_model,crop_status_model])


# Freeze the base_model
#base_model.trainable = False

full_model.compile(optimizer = keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])
print(full_model.summary())

full_model.fit(train_generator,
         # validation_data= (val_x,val_y),
          #class_weight = weights,
          steps_per_epoch=ceil(train_sample_size/batch_size), # this is not automatic cause of custom generator
          #validation_freq = 2,
          epochs=2)
