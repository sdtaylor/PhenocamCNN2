This is the repository for the following study:

Deep learning models for identifying crop and field attributes from near surface cameras. 

# File Structure
## A. Initial Preparation
**phenocam_data_prep/**  
    1. `generate_site_list.R` - with the phenocamr package make a list of cameras to use in the study to `site_list.csv`.
    2. `download_phenocam_data.R` - for each camera download the Gcc and and transition dates to `data/phenocam_gcc/`. 
    3. `generate_training_image_list.R` - for each camera use the Gcc transition dates to partition each calendar year into distinct periods of senesced, growth, peak, and senescing. Randomly choose images among these periods for each camera to make a list of training images. Creates the file `data/images_for_annotation.csv`. Also creates the file `data/full_image_list.csv` which is is the  mid-day image for *every* available day for all sites in `site_list.csv`.
    4. `download_training_phenocam_images.R` - download all images in `data/images_for_annotation.csv`.
    5. `generate_extra_image_list.R` - for each image in `data/images_for_annotation.csv` get the download link of *all* images from the respective day. These are the ~80k additional images taken from 0900-1500 described in the text. Creates the file `data/extra_images_for_fitting.csv`
    6. `download_extra_phenocam_images.R` - download all images in `data/extra_images_for_fitting.csv` to the folder `data/extra_phenocam_train_images/`
    7. `download_all_phenocam_images.R` - download all images in `data/full_image_list.csv` to `data/phenocam_all_images/`
    
## B. Training Image Annotation 
**train_image_annotation/**  
    1. `imageant_config2.ias` - configuration file for the annotation software, imageant.  https://gitlab.com/stuckyb/imageant. Note I used an older version that was is currently availble, and this ias file will not work with the current version. In fact, the one I used is so old it does not have a version number. But it lives at commit 3c2fd39.  
    2. `imageant_session2.csv` - this is a session file for imageant
    3. `image_classes.csv` - pairing for annotation numeric and text labels. (eg. dominant cover class 1 = vegetation).
    4. `merge_new_crop_types.R` - A little needed data munging. See file for details.
    5. `image_annotations.csv` - The  final annotation file from imagant, eg. the file with all the training/validation data labels. This is used in model fitting and final evaluation.

## C. Model Fitting and prediction
    1. `fit_keras_model.py` - VGG16 model fitting. Uses all annotated and "extra" images to fit the model and writes the keras file `data/vgg16_v4_20epochs.h5`.
    2. `apply_keres_model.py` - Using the fitted VGG16 model make predictions on everything in `data/extra_phenocam_train_images/` and `data/phenocam_all_images/`. Writes those predictions to `data/vgg16_v4_20epochs_predictions.csv` 

## D. Postprocessing
**classification_postprocessing/**  
    1. `prep_predictions_for_hmm.R` - takes the file `data/vgg16_v4_20epochs_predictions.csv` and preps the predictions for the HMM aspect. Creates several files `data/image_predictions_for_*`. 
    2. `final_processing.R`- Takes output from the  HMM model (`./data/hmm_output.csv`) and applies the final post-processing steps (see text) producing `./data/final_predictions.csv`.
    3. `postprocessing_tools.R` - helper functions.

## E. Hidden Markov Model (HMM)
    1. `hmm_stuff/hmm_model_definitions.py` - This describes the HMM models using the pomegranate package. https://pomegranate.readthedocs.io.  
    2. `apply_hmm_model.py` - Applies HMM model to the  `data/image_predictions_for_*` files. 

## F. Analayis
**analysis/**  
    1. `evaluate_predictions.R` - calculate error metrcis and create manuscript F1/precision/recall figures. This uses all files in the process:   `data/vgg16_v4_20epochs_predictions.csv`, `image_annotations.csv`, and `./data/final_predictions.csv`.
    2. `timeseries_figures.R` - produces the colorful timeseries figures for each site year.
    3. `site_map_and_table.R` - produces some supplemental material. 
    4. `single_image_diagnostic_plots.R` - produces the supplemental figures where prediction probabilites for single images are displayed.
    
    
# Workflow
The modelling workflow went as so.  
1. Determine the needed images and download all of them (A)
2. Annotate all the images (B).
3. Fit the vgg16 model and predict on the full image dataset (C)
4. Apply post processing (D,E).
    With the VGG16 output post-processing is in the following order:
    1. `prep_predictions_for_hmm.R`
    2. `apply_hmm_model.py`
    3. `final_processing.R`
5. Analize and Visualize (F).
    
    
    
    
