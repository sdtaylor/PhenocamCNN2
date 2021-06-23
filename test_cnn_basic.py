from glob import glob
from os.path import basename
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

from sklearn.model_selection import ParameterGrid

#----------------
# config stuff
#----------------
image_dir = 'data/phenocam_train_images/'
extra_image_dir = 'data/extra_phenocam_train_images/'

train_sample_size = 100000

validation_fraction = 0.2
target_size = (224,224)
batch_size  = 50
unique_category_min_n = 40 # each combo of dominant-cover/crop_type/crop_status must have at least this many original images

image_info = pd.read_csv('train_image_annotation/image_annotations.csv')
extra_image_info = pd.read_csv('data/extra_images_for_fitting.csv')

# whoops, the filename exractor dropped the extension
extra_image_info['filenames'] = extra_image_info.filenames + '.jpg'

# drop extra images which are duplicates
extra_image_info = extra_image_info[~extra_image_info.filenames.isin(image_info.file)]

# drop any missing extra image. Over 100k so a few had failed downloads
available_extra_images = glob(extra_image_dir+'*jpg')
available_extra_images = [basename(i) for i in available_extra_images]
extra_image_info = extra_image_info[extra_image_info.filenames.isin(available_extra_images)]

# some category combinations are not well represented (eg. snow + unknown_plant + senescing)
# so just don't bother with them
image_info['unique_category'] = image_info.dominant_cover.astype(str) + image_info.crop_type.astype(str) + image_info.crop_status.astype(str)
to_keep = image_info.unique_category.value_counts().reset_index().rename(columns={'index':'unique_category','unique_category':'n'})
to_keep['keep'] = to_keep.n >= unique_category_min_n

image_info = image_info.merge(to_keep, on='unique_category', how='left')
image_info = image_info[image_info.keep].drop(columns=['unique_category','n','keep'])

print('working with {} extra images, {} annoated images'.format(extra_image_info.shape[0], image_info.shape[0]))


def extract_phenocam_name(filename):
    # filenames look like 'arsgacp1_2016_07_25_120000.jpg'
    return filename.split('_')[0]

def extract_date(filename):
    filename_split = filename.split('_')
    year  = filename_split[1]
    month = filename_split[2]
    day   = filename_split[3]
    return '{}-{}-{}'.format(year,month,day)

#----------------
# Merge the primary image_info, which was annoated by hand, with the extra_image_info
# which was not annotated but consists of the same sites and dates.
#---------------

image_info['date'] = image_info.file.map(extract_date)
image_info['phenocam_name'] = image_info.file.map(extract_phenocam_name)

image_info['filepath'] = image_dir + image_info.file
extra_image_info['filepath'] = extra_image_dir + extra_image_info.filenames

image_info       = image_info[['filepath','phenocam_name','date','dominant_cover','crop_type','crop_status']]
extra_image_info = extra_image_info[['filepath','phenocam_name','date']]


# Assigned annotations to the extra images
extra_image_info = extra_image_info.merge(image_info[['phenocam_name','date','dominant_cover','crop_type','crop_status']],
                                          how = 'left',
                                          on  = ['phenocam_name','date'])

# Drop any extra images without matching annotations that snuck in
extra_image_info = extra_image_info[~extra_image_info.dominant_cover.isna()]

all_image_info = image_info.append(extra_image_info).reset_index()

#-------------------------
# drop the fallow crop type. There was just not that many of it in the end.
# might be tricky just switching the numbers around though.
#----------------

#TODO

# The different targest and their number of classes
output_classes = {'dominant_cover' : 6,
                  'crop_type'      : 8,
                  'crop_status'    : 7}
#output_classes = {'dominant_cover' : 6}



#-------------------------
# Setup validation split
#----------------
total_validation_images = int(len(all_image_info) * validation_fraction)

# First put all images from held out sites into the validation set
all_image_info['validation_site'] = all_image_info.phenocam_name.apply(lambda f: bool(re.search(r'(arsmorris2)|(mandani2)|(cafboydnorthltar01)', f)))
validation_images = all_image_info[all_image_info.validation_site]

# Add in a random sample of remaining images to get to the total validation fraction
# images from validation sites excluded here by setting weight to 0
all_image_info['validation_weight'] = all_image_info.validation_site.apply(lambda val_site: 0 if val_site else 1)
validation_images = validation_images.append(all_image_info.sample(n= total_validation_images - len(validation_images), replace=False, random_state=99, weights='validation_weight')) 

# Training images are ones that are left
train_images = all_image_info[~all_image_info.index.isin(validation_images.index)]

# assure no validation sites in the training data, and all images in each set are unique
assert train_images.validation_site.sum() == 0, 'validation sites in training dataframe'
assert train_images.index.nunique() == len(train_images), 'duplicates in training dataframe'
assert validation_images.index.nunique() == len(validation_images), 'duplicates in validation dataframe'

#-------------------------
# expand training by random sampling, weighted so that low sample size category images are repeated.
# This makes it so sample sizes are even in training

# Use the unique combinations across the 3 categories. 
train_images['unique_category'] = train_images.dominant_cover.astype(str) + train_images.crop_type.astype(str) + train_images.crop_status.astype(str)
train_images['sample_weight'] = compute_sample_weight('balanced', train_images.unique_category)
train_images = train_images.sample(n=train_sample_size, replace=True, random_state=99, weights='sample_weight')

#-------------------------
# Generate numpy arrays of all images. Each image is repeated several times from the sampling
train_x = load_imgs_from_df(train_images, x_col='filepath', img_dir='', 
                            target_size=target_size, data_format='channels_last')

# Multiple output model means a dictionary for the targets
# these names match the final model layer names
train_y = {c:to_categorical(train_images[c]) for c in output_classes.keys()}


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

# small sample for testing                                         
#validation_images = validation_images.sample(500)                      
val_x =  load_imgs_from_df(validation_images, x_col='filepath', img_dir='', 
                            target_size=target_size, data_format='channels_last')
val_y = {c:to_categorical(validation_images[c]) for c in output_classes.keys()}

#-------------------------
# The keras model
#-------------------------
input_layer = Input(shape=target_size + (3,))
input_layer =  Rescaling(scale = 1./127.5, offset=-1)(input_layer)

base_model = keras.applications.VGG16(
    weights=None,  # Load weights pre-trained on ImageNet.
    #input_shape= target_size + (3,),
    input_tensor = input_layer,
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

def build_category_model(prior_step, class_n, name):
    x = keras.layers.Flatten()(prior_step) 
    x = keras.layers.Dense(4096, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(class_n,  activation = 'softmax', name=name)(x)
    return(x)

sub_models = [build_category_model(base_model.output, class_n, name) for name, class_n in output_classes.items()]

full_model = keras.Model(base_model.input, sub_models)

#-------------------------
# save the initial random weights so they can be reloaded
init_weights_file = 'data/initial_weights.h5'
full_model.save_weights(init_weights_file)

optimizer_param_grid = ParameterGrid({'lr':[0.01,0.001,0.0001],'epsilon':[1.0,0.1,0.01]})

#for i, optimzer_params in enumerate(optimizer_param_grid):
optimzer_params = dict(lr=0.01, epsilon=0.1)

#full_model.load_weights(init_weights_file)
#print(full_model.summary())

# First round of fitting, 20 epochs
full_model.compile(optimizer = keras.optimizers.Adam(lr=optimzer_params['lr'], epsilon=optimzer_params['epsilon']),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])

full_model.fit(train_generator,
               validation_data= (val_x,val_y),
               #class_weight = weights,
               steps_per_epoch=ceil(train_sample_size/batch_size), # this is not automatic cause of custom generator
               epochs=15,
               use_multiprocessing=False)


trace_history1 = pd.DataFrame(full_model.history.history)

# 2nd round of fitting, 5 epochs with lower learning rate
optimzer_params = dict(lr=0.001, epsilon=0.1)
full_model.compile(optimizer = keras.optimizers.Adam(lr=optimzer_params['lr'], epsilon=optimzer_params['epsilon']),
              loss='categorical_crossentropy',metrics=[keras.metrics.CategoricalAccuracy()])

full_model.fit(train_generator,
               validation_data= (val_x,val_y),
               #class_weight = weights,
               steps_per_epoch=ceil(train_sample_size/batch_size), # this is not automatic cause of custom generator
               epochs=5,
               use_multiprocessing=False)

trace_history2 = pd.DataFrame(full_model.history.history)
trace_history1.append(trace_history2).to_csv('data/vgg16_v4_20epochs_trace.csv', index=False)
full_model.save('data/vgg16_v4_20epochs.h5')
