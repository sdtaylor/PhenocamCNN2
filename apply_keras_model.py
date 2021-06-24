import glob
from os.path import basename

import pandas as pd

from tensorflow import keras
from tools import keras_tools

# The different targest and their number of classes
output_classes = {'dominant_cover' : 6,
                  'crop_type'      : 7,
                  'crop_status'    : 7}

results_file = './data/vgg16_v1_55epochs_predictions.csv'

keras_model = keras.models.load_model('./data/vgg16_v1_55epochs.h5')

class_categories = pd.read_csv('train_image_annotation/image_classes.csv')

# fit to all images in phenocam data dump from ORNL as well as all training images
# there are some duplicates which will be dealt with later.
image_dirs = ['/project/ltar_phenology_proj1/PhenocamCNN2/data/PhenocamCNN_images/',
              './data/phenocam_train_images/']

all_images = []
for image_dir in image_dirs:
    all_images.extend(glob.glob(image_dir+ '**/*.jpg', recursive=True))

image_info = pd.DataFrame(dict(filepath = all_images))

image_info['file'] = image_info.applymap(basename)

#image_info = image_info.sample(10)

predictions = keras_tools.keras_predict(df = image_info, filename_col='filepath', 
                                        model = keras_model, target_size=(224,224),
                                        preprocess_func=None)

#----------------------
# Make a dataframe like:
# file, dominant_cover-unknown, dominant_cover-vegetation,...,crop_status-small_grass, crop_status-large_grass,...
# with probabilites throughout.
# The order of the category output in the prediction is the same as  output_classes since 
# the same dictionary is used in the model building and fitting.

for category_i, (category, n_classes) in enumerate(output_classes.items()):
    pass
    category_descriptions = class_categories[class_categories.category==category]
    
    col_headers = []
    for c in category_descriptions.class_description:
        col_header = '{}-{}'.format(category, c)
        image_info[col_header] = 0
        col_headers.append(col_header)

    image_info[col_headers] = predictions[category_i].round(4)

image_info.to_csv(results_file, index=False)
